import torch
import torch.nn as nn

class MultiLabelBCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-5):
        super(MultiLabelBCEDiceLoss, self).__init__()
        self.alpha = alpha  # BCE权重
        self.smooth = smooth
        self.bce_loss = nn.BCELoss()  # 多标签二分类损失

    def soft_dice_coeff(self, y_pred, y_true):
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        return (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)

    def soft_dice_loss(self, y_pred, y_true):
        return 1 - self.soft_dice_coeff(y_pred, y_true)

    def forward(self, y_pred, y_true):
        # y_pred: [B, 6, H, W]，sigmoid输出
        # y_true: [B, 6, H, W]，独热编码（0/1）
        bce = self.bce_loss(y_pred, y_true)
        dice = self.soft_dice_loss(y_pred, y_true)
        return self.alpha * bce + (1 - self.alpha) * dice