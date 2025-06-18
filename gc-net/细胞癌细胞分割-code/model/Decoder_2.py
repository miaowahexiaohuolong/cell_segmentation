import torch
import torch.nn as nn
from Decoder.RCA import RCAblock  # 导入RCA模块
from Decoder.GCA import GCAblock  # 导入GCA模块
from DenseNetBlock import DenseNetEncoder  # 导入DenseNet编码器

class Decoder_1(nn.Module):
    def __init__(self, 
                 rca_in_channels=256,   # RCA模块的输入通道数（匹配Block3的256通道）
                 rca_out_channels=256,  # RCA输出通道数（与GCA输入一致）
                 gca_reduction=4,       # GCA通道降维比例
                 transpose_out_channels=3,  # 转置卷积输出通道数（如RGB）
                 dense_encoder_out_index=2  # DenseNet输出索引（2对应64×64，256通道）
                ):
        super(Decoder_1, self).__init__()
        
        self.dense_encoder = DenseNetEncoder()
        self.dense_encoder_out_index = dense_encoder_out_index  # 提取索引2的输出（64×64）
        
        # 调整通道数的卷积层（输入256通道，输出与RCA一致的256通道，可省略若无需调整）
        self.adapter_conv = nn.Conv2d(
            in_channels=self.get_dense_channels(),
            out_channels=rca_out_channels,
            kernel_size=1
        )
        
        self.rca = RCAblock(
            in_channels=rca_in_channels, 
            out_channels=rca_out_channels
        )
        self.gca = GCAblock(
            in_channels=rca_out_channels, 
            reduction=gca_reduction
        )
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels=rca_out_channels,
            out_channels=transpose_out_channels,
            kernel_size=4,
            stride=2,   # 上采样64→128
            padding=1,
            output_padding=0
        )

    def get_dense_channels(self):
        """获取DenseNet对应输出的通道数（索引2→256通道）"""
        channel_list = [64, 128, 256, 512]  # 索引0~3对应Block1~Block4的通道数
        return channel_list[self.dense_encoder_out_index]

    def forward(self, x_dense, x_rca):
        # ---------------------- 提取DenseNet的64×64中间输出（Block3，256通道） ----------------------
        dense_features = self.dense_encoder(x_dense)
        dense_feat = dense_features[self.dense_encoder_out_index]  # 形状：[b, 256, 64, 64]
        
        # 调整通道数（若需要，此处假设RCA输入为256通道，可省略adapter_conv）
        dense_feat_adapted = self.adapter_conv(dense_feat)  # 输出：[b, 256, 64, 64]
        
        # ---------------------- RCA→GCA分支（处理64×64特征） ----------------------
        rca_feat = self.rca(x_rca)  # x_rca形状：[b, 256, 64, 64] → 输出：[b, 256, 64, 64]
        gca_feat = self.gca(rca_feat)  # GCA输出：[b, 256, 64, 64]
        
        # ---------------------- 残差连接 ----------------------
        residual_feat = gca_feat + dense_feat_adapted  # 形状：[b, 256, 64, 64]
        
        # ---------------------- 转置卷积上采样（64×64→128×128） ----------------------
        output = self.transpose_conv(residual_feat)  # 输出：[b, transpose_out_channels, 128, 128]
        
        return output,residual_feat

# -------------------------- 模型测试 --------------------------
if __name__ == "__main__":
    # 输入形状说明：
    # - x_dense: DenseNet输入，512×512图像
    # - x_rca: RCA输入，64×64特征图，通道数256（与Block3通道一致）
    x_dense = torch.randn(2, 3, 512, 512)
    x_rca = torch.randn(2, 256, 64, 64)  # 注意：此处通道数改为256，尺寸64×64
    
    model = Decoder_1(
        rca_in_channels=256,
        rca_out_channels=256,
        transpose_out_channels=3
    )
    
    output = model(x_dense, x_rca)
    
    # 验证各阶段形状
    print(f"DenseNet Block3输出形状（64×64）: {model.dense_encoder(x_dense)[2].shape}")
    # 输出: torch.Size([2, 256, 64, 64])
    
    print(f"RCA输出形状: {model.rca(x_rca).shape}")
    # 输出: torch.Size([2, 256, 64, 64])
    
    print(f"GCA输出形状: {model.gca(model.rca(x_rca)).shape}")
    # 输出: torch.Size([2, 256, 64, 64])
    
    #print(f"残差连接后形状: {residual_feat.shape}")  # 假设在forward中打印，形状同上
    print(f"最终输出形状: {output.shape}")
    # 输出: torch.Size([2, 3, 128, 128])（64×2=128）