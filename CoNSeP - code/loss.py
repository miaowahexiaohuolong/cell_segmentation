import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------- 多类别交叉熵损失（支持权重） -------------------
class WeightedCrossEntropy(nn.Module):
    def __init__(self, num_classes=8, weight=None):
        super().__init__()
        self.num_classes = num_classes
        
        # 初始化类别权重（默认全1）
        if weight is not None:
            if isinstance(weight, (list, tuple)):
                weight = torch.tensor(weight, dtype=torch.float32)
            if weight.ndim != 1 or weight.shape[0] != num_classes:
                raise ValueError(f"权重需为长度{num_classes}的1维张量，当前形状{weight.shape}")
            self.weight = weight
        else:
            self.weight = torch.ones(num_classes, dtype=torch.float32)
        
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, y_pred, y_true):
        # 调整真实标签维度：[B,1,H,W] → [B,H,W]（CrossEntropyLoss要求）
        if y_true.dim() == 4 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1).long()  # 压缩单通道维度
        else:
            raise ValueError(f"y_true 需为 [B,1,H,W]，当前形状 {y_true.shape}")
        
        return self.ce_loss(y_pred, y_true)


# ------------------- 多类别Dice损失 -------------------
class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # 将真实标签转换为 one-hot 格式 [B, C, H, W]
        num_classes = y_pred.shape[1]
        y_true_flat = y_true.squeeze(1).long()  # [B,1,H,W] → [B,H,W]
        y_true_onehot = F.one_hot(y_true_flat, num_classes=num_classes)  # [B,H,W,C]
        y_true_onehot = y_true_onehot.permute(0, 3, 1, 2).float()  # [B,C,H,W]

        # 计算每个类别的交集和并集
        intersection = torch.sum(y_true_onehot * y_pred, dim=(2, 3))  # [B,C]
        union = torch.sum(y_true_onehot + y_pred, dim=(2, 3))         # [B,C]

        # 计算Dice系数（平均所有类别和批次）
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B,C]
        dice = dice.mean()  # 全局平均

        return 1 - dice  # Dice损失 = 1 - Dice系数


# ------------------- 多类别组合损失（交叉熵+Dice） -------------------
class MultiClassCombinedLoss(nn.Module):
    def __init__(self, num_classes=8, ce_weight=None, dice_smooth=1e-5):
        super().__init__()
        self.ce_loss = WeightedCrossEntropy(num_classes, ce_weight)
        self.dice_loss = MultiClassDiceLoss(dice_smooth)

    def forward(self, y_pred, y_true):
        # 交叉熵损失（输入logits）
        ce = self.ce_loss(y_pred, y_true)
        
        # 计算概率（softmax）
        y_pred_soft = F.softmax(y_pred, dim=1)
        
        # Dice损失（输入概率）
        dice = self.dice_loss(y_pred_soft, y_true)
        
        return ce + dice


# ------------------- 可选损失函数（根据需求使用） -------------------
class DiceLoss(nn.Module):
    """ 单类别Dice损失（二值分割用） """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = input.sigmoid()
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        union = input_flat.sum() + target_flat.sum()
        return 1 - (2. * intersection + self.smooth) / (union + self.smooth)


class FocalLoss(nn.Module):
    """ 焦点损失（适用于类别不平衡） """
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        target = target.squeeze(1).long()
        ce_loss = F.cross_entropy(input, target, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def pixel_accuracy(y_pred, y_true):
    """
    计算像素级分类准确率
    Args:
        y_pred: 模型输出logits，形状[B, C, H, W]（C=num_classes）
        y_true: 真实标签，形状[B, 1, H, W]（单通道，数值0-7）
    Returns:
        准确率（正确像素数 / 总像素数）
    """
    # 预测类别：[B, C, H, W] → [B, H, W]（取概率最大的类别）
    pred_classes = torch.argmax(F.softmax(y_pred, dim=1), dim=1)  # [B, H, W]
    
    # 真实标签：[B, 1, H, W] → [B, H, W]
    true_classes = y_true.squeeze(1).long()  # [B, H, W]
    
    # 计算正确像素数
    correct = torch.sum(pred_classes == true_classes)
    total = torch.numel(true_classes)  # 总像素数
    
    return correct / total
# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 模拟输入：B=2, C=8, H=448, W=448
    y_pred = torch.randn(2, 8, 448, 448)
    y_true = torch.randint(0, 8, (2, 1, 448, 448))  # 单通道标签（0-7类）

    loss_fn = MultiClassCombinedLoss(num_classes=8)
    total_loss = loss_fn(y_pred, y_true)
    print(f"总损失: {total_loss.item():.4f}")  # 应输出合理数值（非nan/inf）