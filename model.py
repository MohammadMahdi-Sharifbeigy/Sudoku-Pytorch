import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights,
)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss


class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


class MultiTaskDigitCNN(nn.Module):
    """
    Shared 2-conv backbone (identical to DigitCNN) with two prediction heads.

    digit_head : Linear(3136,32) -> ReLU -> Dropout(0.5) -> Linear(32,10)
                 classes: 0=empty, 1-9=digit
    lang_head  : Linear(3136,16) -> ReLU -> Linear(16,2)
                 classes: 0=Persian, 1=English

    Empty cells must carry lang_target=-1 so CrossEntropyLoss(ignore_index=-1)
    masks them from language loss.

    Total parameters: ~169,756 (well under 200k).
    forward() returns (digit_logits, lang_logits).
    """
    def __init__(self, num_digit_classes=10, num_lang_classes=2):
        super(MultiTaskDigitCNN, self).__init__()
        # Shared backbone — same conv blocks as DigitCNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()  # 64 * 7 * 7 = 3136

        # Digit head (hidden=32, 10 classes)
        self.digit_fc1  = nn.Linear(3136, 32)
        self.digit_relu = nn.ReLU(inplace=True)
        self.digit_drop = nn.Dropout(p=0.5)
        self.digit_fc2  = nn.Linear(32, num_digit_classes)

        # Language head (hidden=16, binary)
        self.lang_fc1  = nn.Linear(3136, 16)
        self.lang_relu = nn.ReLU(inplace=True)
        self.lang_fc2  = nn.Linear(16, num_lang_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        digit_logits = self.digit_fc2(self.digit_drop(self.digit_relu(self.digit_fc1(x))))
        lang_logits  = self.lang_fc2(self.lang_relu(self.lang_fc1(x)))
        return digit_logits, lang_logits


class MultiTaskFocalLoss(nn.Module):
    """
    Combined loss for MultiTaskDigitCNN.

    total = digit_weight * FocalLoss(digit_logits, digit_target)
          + lang_weight  * CrossEntropyLoss(lang_logits, lang_target, ignore_index=-1)

    lang_class_weights (shape 2,) corrects Persian/English imbalance via
    inverse-frequency weighting — computed by data_utils.compute_lang_class_weights.

    Returns (total_loss_tensor, digit_loss_float, lang_loss_float).
    """
    def __init__(self, digit_weight=0.7, lang_weight=0.3,
                 focal_alpha=0.25, focal_gamma=2.0,
                 lang_class_weights=None, lang_ignore_index=-1):
        super(MultiTaskFocalLoss, self).__init__()
        self.digit_weight    = digit_weight
        self.lang_weight     = lang_weight
        self.lang_ignore_idx = lang_ignore_index
        self.focal   = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.lang_ce = nn.CrossEntropyLoss(
            weight=lang_class_weights,
            ignore_index=lang_ignore_index,
        )

    def forward(self, digit_logits, lang_logits, digit_target, lang_target):
        d_loss = self.focal(digit_logits, digit_target)

        # Guard: if all lang_targets are the ignore sentinel (e.g. an all-empty-cell
        # batch), CrossEntropyLoss(reduction='mean') returns nan (0 valid / 0 count).
        # Return zero lang loss instead so the total stays finite.
        valid_lang = (lang_target != self.lang_ignore_idx)
        if valid_lang.any():
            l_loss = self.lang_ce(lang_logits, lang_target)
        else:
            l_loss = torch.zeros(1, device=digit_logits.device, dtype=digit_logits.dtype).squeeze()

        total = self.digit_weight * d_loss + self.lang_weight * l_loss
        return total, d_loss.item(), l_loss.item()


# ──────────────────────────────────────────────
# Unified 20-class model  (digit × language)
# ──────────────────────────────────────────────

# Class layout:
#   0        → English 0  (empty cell)
#   1 – 9    → English 1-9
#   10       → Persian 0  (empty cell)
#   11 – 19  → Persian 1-9
UNIFIED_NUM_CLASSES = 20


def decode_unified_class(cls_idx: int):
    """Map a 20-class prediction to (digit 0-9, lang_name or None).

    Returns
    -------
    digit    : int  0 = empty, 1-9 = digit value
    lang     : str | None  'English', 'Persian', or None for empty cells
    """
    if cls_idx == 0:
        return 0, None          # empty (English source)
    if cls_idx == 10:
        return 0, None          # empty (Persian source)
    if cls_idx < 10:
        return cls_idx, 'English'
    return cls_idx - 10, 'Persian'


class UnifiedDigitCNN(nn.Module):
    """
    20-class unified model: classifies digit (0-9) AND script in one pass.

    Backbone choices
    ----------------
    'mobilenet_v3_small' — ~2.5 M params (pretrained=False recommended for 28×28)
    'shufflenet_v2_x0_5' — ~0.35 M params (lightest option)

    First conv adapted from 3-channel to 1-channel (grayscale 28×28 input).
    Final classifier replaced with Linear(→ num_classes=20).

    forward(x) returns logits of shape (B, 20).
    Use decode_unified_class() to map predicted class to (digit, lang).
    """

    def __init__(self, backbone: str = 'mobilenet_v3_small',
                 pretrained: bool = False,
                 num_classes: int = UNIFIED_NUM_CLASSES):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes   = num_classes

        if backbone == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = mobilenet_v3_small(weights=weights)
            # Adapt first conv: 3 → 1 channel
            old = net.features[0][0]
            net.features[0][0] = nn.Conv2d(
                1, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias is not None,
            )
            # Replace classifier head
            in_feat = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_feat, num_classes)
            self.net = net

        elif backbone == 'shufflenet_v2_x0_5':
            weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
            net = shufflenet_v2_x0_5(weights=weights)
            # Adapt first conv: 3 → 1 channel
            old = net.conv1[0]
            net.conv1[0] = nn.Conv2d(
                1, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias is not None,
            )
            # Replace fc
            net.fc = nn.Linear(net.fc.in_features, num_classes)
            self.net = net

        else:
            raise ValueError(f"Unknown backbone '{backbone}'. "
                             "Choose 'mobilenet_v3_small' or 'shufflenet_v2_x0_5'.")

    def forward(self, x):
        return self.net(x)

    def param_count(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable