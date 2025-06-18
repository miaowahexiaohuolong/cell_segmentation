import torch
import torch.nn as nn

class RCAblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCAblock, self).__init__()
        # 1×1 卷积分支
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1x1 = nn.BatchNorm2d(out_channels)
        # 3×3 卷积分支
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn3x3_1 = nn.BatchNorm2d(out_channels)
        self.relu_mid = nn.ReLU()
        self.conv3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3x3_2 = nn.BatchNorm2d(out_channels)
        self.final_relu = nn.ReLU()

    def forward(self, x):
        # 1×1 卷积分支
        branch1 = self.conv1x1(x)
        branch1 = self.bn1x1(branch1)
        # 3×3 卷积分支
        branch2 = self.conv3x3_1(x)
        branch2 = self.bn3x3_1(branch2)
        branch2 = self.relu_mid(branch2)
        branch2 = self.conv3x3_2(branch2)
        branch2 = self.bn3x3_2(branch2)
        # 特征融合与最终激活
        combined = branch1 + branch2  # 残差连接方式融合特征
        out = self.final_relu(combined)
        return out

# 示例用法
if __name__ == "__main__":
    input_tensor = torch.randn(2, 64, 32, 32)  # 模拟输入：2 张图，64 通道，32×32 尺寸
    rca = RCAblock(in_channels=64, out_channels=64)
    output = rca(input_tensor)
    print(f"输入形状: {input_tensor.shape}")  # 输出: torch.Size([2, 64, 32, 32])
    print(f"输出形状: {output.shape}")  # 输出: torch.Size([2, 64, 32, 32])