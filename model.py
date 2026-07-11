import torch
import torch.nn as nn


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
        self.digit_weight = digit_weight
        self.lang_weight  = lang_weight
        self.focal   = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.lang_ce = nn.CrossEntropyLoss(
            weight=lang_class_weights,
            ignore_index=lang_ignore_index,
        )

    def forward(self, digit_logits, lang_logits, digit_target, lang_target):
        d_loss = self.focal(digit_logits, digit_target)
        l_loss = self.lang_ce(lang_logits, lang_target)
        total  = self.digit_weight * d_loss + self.lang_weight * l_loss
        return total, d_loss.item(), l_loss.item()