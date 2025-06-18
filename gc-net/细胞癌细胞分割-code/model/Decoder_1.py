import torch
import torch.nn as nn
from Decoder.RCA import RCAblock  # 假设RCAblock支持输出通道数调整
from Decoder.GCA import GCAblock  # GCA输出通道数等于输入通道数（由GCAblock代码保证）
from DenseNetBlock import DenseNetEncoder

class Decoder_1(nn.Module):
    def __init__(self, 
                 rca_in_channels=36,    # RCA输入通道数（与x_rca的36通道匹配）
                 rca_out_channels=512,  # RCA输出通道数（与DenseNet的512通道一致）
                 gca_reduction=4,       # GCA模块的通道降维比例（不影响输出通道数）
                 transpose_out_channels=3,  # 转置卷积的输出通道数（如RGB图像）
                 dense_encoder_out_index=3  # DenseNet输出索引（取第4个输出，通道512）
                ):
        super(Decoder_1, self).__init__()
        
        self.dense_encoder = DenseNetEncoder()
        self.dense_encoder_out_index = dense_encoder_out_index  # 提取第几个中间输出（0~3）
        
        # 移除通道适配层（self.channel_adapter）
        
        # RCA输出通道数调整为512（与DenseNet中间输出通道一致）
        self.rca = RCAblock(in_channels=rca_in_channels, out_channels=rca_out_channels)
        # GCA输入通道数调整为512（与RCA输出一致，输出通道数也为512）
        self.gca = GCAblock(in_channels=rca_out_channels, reduction=gca_reduction)
        # 转置卷积输入通道数调整为512（与残差连接后的通道一致）
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels=rca_out_channels,  # 改为512
            out_channels=transpose_out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0
        )

    def forward(self, x_dense, x_rca):
        # 提取DenseNet的中间输出（通道数512）
        dense_features = self.dense_encoder(x_dense)  # 输入x_dense：[B, 3, 512, 512]
        dense_feat = dense_features[self.dense_encoder_out_index]  # 形状：[B, 512, 32, 32]
        
        # RCA处理（输入36通道→输出512通道）
        rca_feat = self.rca(x_rca)  # 输入x_rca：[B, 36, 32, 32] → 输出[B, 512, 32, 32]
        
        # GCA处理（输入512通道→输出512通道，由GCAblock代码保证）
        gca_feat = self.gca(rca_feat)  # 输出形状：[B, 512, 32, 32]
        
        # 残差连接（通道已匹配，直接相加）
        residual_feat = gca_feat + dense_feat  # 形状：[B, 512, 32, 32]
        
        # 上采样输出（输入512通道→输出目标通道数）
        output = self.transpose_conv(residual_feat)  # 形状：[B, 3, 64, 64]
        
        return output

# 测试代码（与原用例兼容）
if __name__ == "__main__":
    # 输入保持原用例的形状（无需修改）
    x_dense = torch.randn(1, 512, 32, 32)  # DenseNet输入（512通道，32×32）
    x_rca = torch.randn(1, 36, 32, 32)     # RCA输入（36通道，32×32）
    
    # 初始化模型时调整rca_out_channels为512（匹配DenseNet输出通道）
    model = Decoder_1(
        rca_in_channels=36,
        rca_out_channels=512,  # 关键修改：与DenseNet中间输出通道一致
        transpose_out_channels=3
    )
    
    output = model(x_dense, x_rca)
    
    # 验证各阶段形状（无适配层）
    print(f"DenseNet Block4输出形状: {model.dense_encoder(x_dense)[3].shape}")  # 输出: torch.Size([1, 512, 32, 32])
    print(f"RCA输出形状: {model.rca(x_rca).shape}")  # 输出: torch.Size([1, 512, 32, 32])
    print(f"GCA输出形状: {model.gca(model.rca(x_rca)).shape}")  # 输出: torch.Size([1, 512, 32, 32])
    print(f"残差连接后形状: {(model.gca(model.rca(x_rca)) + model.dense_encoder(x_dense)[3]).shape}")  # 输出: torch.Size([1, 512, 32, 32])
    print(f"最终输出形状: {output.shape}")  # 输出: torch.Size([1, 3, 64, 64])