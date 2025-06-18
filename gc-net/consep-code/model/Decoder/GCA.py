import torch
import torch.nn as nn
import torch.nn.functional as F

class GCAblock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(GCAblock, self).__init__()
        self.in_channels = in_channels  # 输入通道数 c
        self.reduction = reduction      # 降维比例 r
        self.inter_channels = in_channels // reduction  # 中间通道数 c/r
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=1)  
        # 1×1 卷积降维：[b, c, 1, 1] → [b, c/r, 1, 1]
        self.conv_down = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        # LayerNorm：作用于 [c/r, 1, 1] 维度（通道+空间）
        self.ln = nn.LayerNorm((self.inter_channels, 1, 1))
        # ReLU 激活
        self.relu = nn.ReLU()
        # 1×1 卷积恢复维度：[b, c/r, 1, 1] → [b, c, 1, 1]（或自定义输出通道）
        self.conv_up = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        hw = h * w
        # 应该是有三个部分的 一个是最左边，一个是中间，一个是右边

        # 左边分支
        origon = x

        # ---------------------- 右边分支：空间注意力 ----------------------
        # 1. 1×1 卷积：压缩通道到 1（形状：[b, 1, h, w]）
        spatial_weights = self.spatial_conv(x)  

        # 2. 调整维度顺序：[b, 1, h, w] → [b, h, w, 1]（便于展平空间维度）
        spatial_weights = spatial_weights.permute(0, 2, 3, 1)  

        # 3. 展平空间维度：[b, h, w, 1] → [b, hw, 1]（h×w 展平为 hw）
        spatial_weights = spatial_weights.reshape(b, hw, 1)  

        # 4. 添加维度：[b, hw, 1] → [b, hw, 1, 1]（符合 hw×1×1 的要求）
        spatial_weights = spatial_weights.unsqueeze(2)  

        # 5. 对 hw 维度做 softmax：生成归一化权重（形状保持 [b, hw, 1, 1]）
        spatial_weights = F.softmax(spatial_weights, dim=1)

        # ---------------------- 中间分支： 结合起来 ----------------------
        middle_input = x # 【b, c, h, w】

            # 验证空间权重形状
        assert spatial_weights.shape == (b, hw, 1, 1), "空间权重形状错误"
        
        # 展平原始特征图的空间维度：[b, c, h, w] → [b, c, hw]
        x_flat = x.view(b, c, hw)
        
        # 展平空间权重：[b, hw, 1, 1] → [b, hw, 1]
        weights_flat = spatial_weights.view(b, hw, 1)
    
        # 矩阵乘法：[b, c, hw] × [b, hw, 1] → [b, c, 1]
        aggregated = torch.bmm(x_flat, weights_flat)
    
        # 增加单例维度，形状变为 [b, c, 1, 1]
        aggregated = aggregated.unsqueeze(3)

        #1×1卷积降维 → [b, c/r, 1, 1]
        x = self.conv_down(aggregated)
        
        #LayerNorm → 形状不变 [b, c/r, 1, 1]
        x = self.ln(x)
        
        #ReLU 激活 → 形状不变 [b, c/r, 1, 1]
        x = self.relu(x)
        
        #1×1卷积恢复维度 → [b, c, 1, 1]
        x = self.conv_up(x)

        # ---------------------- 在和初始的x拼接计算起来 ----------------------
        output = origon + x
        return output


# 示例用法
if __name__ == "__main__":
    input_tensor = torch.randn(2, 36, 32, 32)  # 模拟输入：2 张图，36 通道，32×32 尺寸
    gca = GCAblock(in_channels=36)
    output = gca(input_tensor)
    print(f"输入形状: {input_tensor.shape}")  # 输出: torch.Size([2, 64, 32, 32])
    print(f"输出形状: {output.shape}")  # 输出: torch.Size([2, 64, 32, 32])
    