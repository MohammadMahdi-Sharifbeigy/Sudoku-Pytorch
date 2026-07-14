import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights,
)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction
        self.ce        = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss    = self.ce(inputs, targets)
        pt         = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DigitCNN(nn.Module):
    """Enhanced CNN for 28×28 grayscale digit recognition.

    Architecture: 3 conv blocks with BatchNorm + residual-style skip,
    followed by a 2-layer MLP head. Significantly stronger than the
    original 2-conv variant while staying under ~250 K parameters.

    Input:  (B, 1, 28, 28)  — normalised with MNIST_MEAN / MNIST_STD
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: 1 → 32  (28×28 → 14×14)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.1),
        )

        # Block 2: 32 → 64  (14×14 → 7×7)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.1),
        )

        # Block 3: 64 → 128  (7×7 → 1×1 via GlobalAvgPool)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),   # GlobalAvgPool: any spatial → 128×1×1
        )

        # Classifier head: 128 → 256 → num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


class LegacyDigitCNN(nn.Module):
    """Original 2-conv architecture (pre-block/BatchNorm redesign).

    Kept only so older checkpoints (state_dict keys: conv1, conv2, fc)
    can still be loaded and exported. Do NOT use for new training —
    use DigitCNN instead.

    Input:  (B, 1, 28, 28)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


def load_digit_cnn_checkpoint(path: str, device, num_classes: int = 10):
    """Load a DigitCNN checkpoint, falling back to LegacyDigitCNN if the
    state_dict doesn't match the current architecture.

    Returns (model, architecture_name) where architecture_name is
    'DigitCNN' or 'LegacyDigitCNN'.
    """
    state_dict = torch.load(path, map_location=device)

    model = DigitCNN(num_classes=num_classes).to(device)
    try:
        model.load_state_dict(state_dict)
        model.eval()
        return model, 'DigitCNN'
    except RuntimeError:
        pass

    legacy = LegacyDigitCNN(num_classes=num_classes).to(device)
    legacy.load_state_dict(state_dict)
    legacy.eval()
    return legacy, 'LegacyDigitCNN'


class MultiTaskDigitCNN(nn.Module):
    """Shared 3-block backbone with two prediction heads.

    digit_head : 128 → 256 → BN → ReLU → Dropout(0.4) → num_digit_classes
    lang_head  : 128 → 64  → ReLU → num_lang_classes

    Empty cells must carry lang_target = -1 so CrossEntropyLoss(ignore_index=-1)
    masks them from the language loss.
    forward() → (digit_logits, lang_logits)
    """

    def __init__(self, num_digit_classes: int = 10, num_lang_classes: int = 2):
        super().__init__()

        # Shared backbone (same as DigitCNN blocks 1-3)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.1),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),   # GlobalAvgPool → 128×1×1
        )
        self.flatten = nn.Flatten()   # 128

        # Digit head: 128 → 256 → num_digit_classes
        self.digit_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_digit_classes),
        )

        # Language head: 128 → 64 → num_lang_classes
        self.lang_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_lang_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feat         = self.flatten(x)
        digit_logits = self.digit_head(feat)
        lang_logits  = self.lang_head(feat)
        return digit_logits, lang_logits


class MultiTaskFocalLoss(nn.Module):
    """Combined loss for MultiTaskDigitCNN.

    total = digit_weight * FocalLoss(digit_logits, digit_target)
          + lang_weight  * CrossEntropyLoss(lang_logits, lang_target, ignore_index=-1)

    Returns (total_loss_tensor, digit_loss_float, lang_loss_float).
    """

    def __init__(self, digit_weight: float = 0.7, lang_weight: float = 0.3,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 lang_class_weights=None, lang_ignore_index: int = -1):
        super().__init__()
        self.digit_weight    = digit_weight
        self.lang_weight     = lang_weight
        self.lang_ignore_idx = lang_ignore_index
        self.focal   = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.lang_ce = nn.CrossEntropyLoss(weight=lang_class_weights,
                                           ignore_index=lang_ignore_index)

    def forward(self, digit_logits, lang_logits, digit_target, lang_target):
        d_loss = self.focal(digit_logits, digit_target)

        valid_lang = (lang_target != self.lang_ignore_idx)
        if valid_lang.any():
            l_loss = self.lang_ce(lang_logits, lang_target)
        else:
            l_loss = torch.zeros(1, device=digit_logits.device,
                                 dtype=digit_logits.dtype).squeeze()

        total = self.digit_weight * d_loss + self.lang_weight * l_loss
        return total, d_loss.item(), l_loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# Unified 20-class model
# ──────────────────────────────────────────────────────────────────────────────
#   0        → English empty cell
#   1 – 9    → English digits 1-9
#   10       → Persian empty cell
#   11 – 19  → Persian digits 1-9

UNIFIED_NUM_CLASSES = 20


def decode_unified_class(cls_idx: int):
    """Map a 20-class prediction → (digit 0-9, lang_name or None)."""
    if cls_idx == 0:   return 0, None
    if cls_idx == 10:  return 0, None
    if cls_idx < 10:   return cls_idx, 'English'
    return cls_idx - 10, 'Persian'


class UnifiedDigitCNN(nn.Module):
    """20-class unified model: digit (0-9) AND script in one pass.

    Backbones:
      'mobilenet_v3_small'  — ~2.5 M params
      'shufflenet_v2_x0_5'  — ~0.35 M params

    First conv adapted 3-ch → 1-ch for grayscale 28×28 input.
    """

    def __init__(self, backbone: str = 'mobilenet_v3_small',
                 pretrained: bool = False,
                 num_classes: int = UNIFIED_NUM_CLASSES):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes   = num_classes

        if backbone == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net     = mobilenet_v3_small(weights=weights)
            old     = net.features[0][0]
            net.features[0][0] = nn.Conv2d(
                1, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias is not None,
            )
            in_feat = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_feat, num_classes)
            self.net = net

        elif backbone == 'shufflenet_v2_x0_5':
            weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
            net     = shufflenet_v2_x0_5(weights=weights)
            old     = net.conv1[0]
            net.conv1[0] = nn.Conv2d(
                1, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias is not None,
            )
            net.fc = nn.Linear(net.fc.in_features, num_classes)
            self.net = net

        else:
            raise ValueError(f"Unknown backbone '{backbone}'. "
                             "Choose 'mobilenet_v3_small' or 'shufflenet_v2_x0_5'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ──────────────────────────────────────────────────────────────────────────────
# EfficientNet-B0 digit model (improved  approach)
# ──────────────────────────────────────────────────────────────────────────────
#   Input:  (B, 1, 28, 28)  grayscale — upsampled to 3-ch inside forward()
#   Output: (B, num_classes) logits
#
#   Two-phase training:
#     Phase 1 — head only, BN frozen  → call freeze_backbone()
#     Phase 2 — last 3 feature blocks + head, BN still frozen → call unfreeze_last_blocks(n=3)
#   BN stays frozen in both phases (correct for pretrained transfer learning).

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetDigitCNN(nn.Module):
    """EfficientNet-B0 adapted for grayscale 28×28 digit recognition.

    Improvements over EfficientNet-B1:
    - B0 (smaller, faster, easier to overfit-guard on small datasets)
    - Grayscale upsampled to 3-ch inside forward() so pretrained conv1 weights apply
    - Head: 1280 → 512 → Dropout(0.3) → 256 → Dropout(0.2) → num_classes
    - freeze_backbone / unfreeze_last_blocks helpers for two-phase training
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = efficientnet_b0(weights=weights)

        # Replace classifier head
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1280, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1, inplace=True),
            nn.Linear(256, num_classes),
        )
        self.net = net
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample 1-ch grayscale → 3-ch so pretrained stem weights are used properly
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.net(x)

    def freeze_backbone(self) -> None:
        """Phase 1: freeze all feature layers, train head only. Freeze BN."""
        for param in self.net.features.parameters():
            param.requires_grad = False
        for param in self.net.classifier.parameters():
            param.requires_grad = True
        self._freeze_batchnorm()

    def unfreeze_last_blocks(self, n: int = 3) -> None:
        """Phase 2: unfreeze last n feature blocks + head. BN stays frozen."""
        blocks = list(self.net.features)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        self._freeze_batchnorm()

    def _freeze_batchnorm(self) -> None:
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable