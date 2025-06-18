import torch
import torch.nn as nn

class MultiKernelBlock(nn.Module):
    def __init__(self, in_channels, out_channels_per_group):
        super(MultiKernelBlock, self).__init__()
        self.groups = nn.ModuleList()
        kernels = [3, 5, 7, 9]  # 卷积核大小
        for kernel in kernels:
            padding = kernel // 2  # 计算填充保持尺寸不变
            group = nn.Sequential(
                nn.Conv2d(in_channels, out_channels_per_group, kernel_size=kernel, padding=padding),
                nn.Conv2d(out_channels_per_group, out_channels_per_group, kernel_size=kernel, padding=padding),
                nn.Sigmoid()
            )
            self.groups.append(group)
    
    def forward(self, x):
        outputs = []
        for group in self.groups:
            out = group(x)
            outputs.append(out)
        # 在通道维度拼接所有组的输出
        return torch.cat(outputs, dim=1)

# 示例用法
if __name__ == "__main__":
    # 模拟输入：批次大小为 2，3 个输入通道，64×64 尺寸
    input_tensor = torch.randn(2, 3, 64, 64)
    # 初始化模块，每个组输出 32 通道
    model = MultiKernelBlock(in_channels=3, out_channels_per_group=32)
    # out_channels = 也就是输出通道，是自己可以设置的
    #print(model)
    output,outputs = model(input_tensor)
    #print(len(outputs)) # 返回四个值，也就是四个卷积
    # 输出形状应为 (2, 4×32, 64, 64) = (2, 128, 64, 64)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")