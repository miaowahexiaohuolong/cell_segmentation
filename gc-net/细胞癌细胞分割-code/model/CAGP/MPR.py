import torch
import torch.nn as nn

class MRPBlock(nn.Module):
    def __init__(self, in_channels, out_channels_per_pool=1):
        """
        初始化 MRP 模块
        :param in_channels: 输入特征图的通道数
        :param out_channels_per_pool: 每个池化分支 1×1 卷积后的输出通道数，默认设为 1（四个分支共 4 通道）
        """
        super(MRPBlock, self).__init__()
        pool_sizes = [2, 4, 6, 8]  # 四种不同大小的池化核
        self.pool_convs = nn.ModuleList()
        for size in pool_sizes:
            self.pool_convs.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((None, None)),  # 自适应平均池化，保持输出尺寸与输入一致
                    nn.Conv2d(in_channels, out_channels_per_pool, kernel_size=1)  # 1×1 卷积降维
                )
            )

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征图，形状为 (batch_size, in_channels, height, width)
        :return: 拼接后的特征图，形状为 (batch_size, in_channels + 4×out_channels_per_pool, height, width)
        """
        original_features = x  # 保留原始特征
        pooled_features = []
        for pool_conv in self.pool_convs:
            pooled = pool_conv(x)
            pooled_features.append(pooled)
        # 在通道维度拼接原始特征和池化后的特征
        output = torch.cat([original_features] + pooled_features, dim=1)
        return output

# 示例用法
if __name__ == "__main__":
    # 模拟输入：批次大小为 2，通道数为 512，尺寸为 32×32
    input_tensor = torch.randn(2, 512, 32, 32)
    mrp = MRPBlock(in_channels=512, out_channels_per_pool=1)
    output = mrp(input_tensor)
    print(f"输入形状: {input_tensor.shape}")  # 应输出: torch.Size([2, 512, 32, 32])
    print(f"输出形状: {output.shape}")  # 应输出: torch.Size([2, 516, 32, 32])（512 + 4×1 = 516）