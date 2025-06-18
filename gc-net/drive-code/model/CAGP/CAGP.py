import torch
import torch.nn as nn
from model.CAGP.MPR import MRPBlock   # 同目录下的MPR.py
from model.CAGP.CGR import MultiKernelBlock  # 同目录下的CGR.py

class CAGP(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, mrp_out_per_pool=1):
        super(CAGP, self).__init__()
        # 第一个 Conv3x3 + BatchNorm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # CGR Block（MultiKernelBlock）
        self.cgr_block = MultiKernelBlock(out_channels, out_channels)
        # 第二个 Conv3x3 + BatchNorm
        self.conv2 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1)  # 4 个组，每组 out_channels 通道
        self.bn2 = nn.BatchNorm2d(out_channels)
        # MRP Block
        self.mrp_block = MRPBlock(out_channels, mrp_out_per_pool)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.cgr_block(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mrp_block(x)
        return x

# 示例用法
if __name__ == "__main__":
    input_tensor = torch.randn(1, 512, 32, 32)  # 模拟输入：1 张图，512 通道，32×32 尺寸
    model = CAGP(in_channels=512, out_channels=32, mrp_out_per_pool=1)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")  # 输出: torch.Size([2, 3, 64, 64])
    print(f"输出形状: {output.shape}")  # 输出: torch.Size([2, 32 + 4×1, 64, 64]) = torch.Size([2, 36, 64, 64])