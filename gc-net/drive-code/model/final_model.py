import torch
import torch.nn as nn
from model.CAGP.CAGP import CAGP
from model.Decoder.RCA import RCAblock
from model.Decoder.GCA import GCAblock
from model.DenseNetBlock import DenseNetEncoder

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        # DenseNet编码器
        self.densenet_encoder = DenseNetEncoder()
        # CAGP模块
        self.cagp = CAGP(in_channels=512, out_channels=32)
        # RCA模块
        self.rca_block1 = RCAblock(in_channels=32 + 4 * 1, out_channels=32 + 4 * 1)
        # GCA模块
        self.gca_block1 = GCAblock(in_channels=32 + 4 * 1)
        # 关键新增：1×1卷积适配层（将512通道降维到36通道）
        self.channel_adapter1 = nn.Conv2d(512, 36, kernel_size=1)
        self.transpose_conv1 = nn.ConvTranspose2d(
            in_channels=36,       # 输入通道数（残差连接后的通道数）
            out_channels=36,      # 输出通道数（可自定义，此处保持一致）
            kernel_size=4,        # 卷积核大小
            stride=2,             # 步长（控制上采样倍数）
            padding=1,            # 填充（保持尺寸计算正确）
            output_padding=0      # 输出填充（可选）
        )

        self.rca_block2 = RCAblock(in_channels=32 + 4 * 1, out_channels=32 + 4 * 1)
        self.gca_block2 = GCAblock(in_channels=32 + 4 * 1)
        self.channel_adapter2 = nn.Conv2d(256, 36, kernel_size=1)
        # 上采样层（32x32 → 64x64）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.transpose_conv2 = nn.ConvTranspose2d(
            in_channels=36,       # 输入通道数（残差连接后的通道数）
            out_channels=36,      # 输出通道数（可自定义，此处保持一致）
            kernel_size=4,        # 卷积核大小
            stride=2,             # 步长（控制上采样倍数）
            padding=1,            # 填充（保持尺寸计算正确）
            output_padding=0      # 输出填充（可选）
        )

        self.rca_block3 = RCAblock(in_channels=32 + 4 * 1, out_channels=32 + 4 * 1)
        self.gca_block3 = GCAblock(in_channels=32 + 4 * 1)
        self.channel_adapter3 = nn.Conv2d(128, 36, kernel_size=1)
        # 上采样层（32x32 → 64x64）
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.transpose_conv3 = nn.ConvTranspose2d(
            in_channels=36,       # 输入通道数（残差连接后的通道数）
            out_channels=36,      # 输出通道数（可自定义，此处保持一致）
            kernel_size=4,        # 卷积核大小
            stride=2,             # 步长（控制上采样倍数）
            padding=1,            # 填充（保持尺寸计算正确）
            output_padding=0      # 输出填充（可选）
        )

        self.channel_adapter4 = nn.Conv2d(64, 36, kernel_size=1)

        self.conv_bn = nn.Sequential(
            nn.Conv2d(36, 36, kernel_size=3, padding=1),
            nn.BatchNorm2d(36)
        )
        self.sigmoid = nn.Sigmoid()

        # 新增上采样层，将输出尺寸调整为 (batch_size, 1, 512, 512)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear')  # 256→512
        self.final_conv = nn.Conv2d(36, 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)  # 对通道维度（dim=1）做Softmax

    def forward(self, x):
        # 通过DenseNet获取特征
        densenet_features = self.densenet_encoder(x)

        #---------------------decoder模块中的各个输出--------------------
        #print("#--------------------decoder模块中的各个输出--------------------")
        # 取最后一个block的输出（32x32尺寸）
        densenet_input1 = densenet_features[-1]
        #print("第一个decoder中的DenseNet输出形状:", densenet_input1.shape) # torch.Size([2, 512, 32, 32])
        # 取第三个block的输出（64x64尺寸）
        densenet_input2 = densenet_features[-2]
        #print("第二个decoder中的DenseNet输出形状:", densenet_input2.shape) # torch.Size([2, 256, 64, 64])
        # 取第二个block的输出（128x128尺寸）
        densenet_input3 = densenet_features[-3]
        #print("第三个decoder中的DenseNet输出形状:", densenet_input3.shape) # torch.Size([2, 128, 128, 128])
        # 取第一个block的输出（256x256尺寸）
        densenet_input4 = densenet_features[0]
        #print("第四个decoder中的DenseNet输出形状:", densenet_input4.shape) # torch.Size([2, 64, 256, 256])

        # 通过CAGP模块
        cagp_output = self.cagp(densenet_input1)
        #print("通过CAGP模块的输出形状",cagp_output.shape) # torch.Size([2, 36, 32, 32])
        #--------------------开始通过第一个decoder模块--------------------
        # 通过RCA模块
        #print("--------------开始通过第一个decoder模块-----------------")
        rca_output1 = self.rca_block1(cagp_output)
        #print("第一个decoder中通过RCA模块的输出形状",rca_output1.shape) #  torch.Size([2, 36, 32, 32])
        # 通过GCA模块
        final_output1 = self.gca_block1(rca_output1)
        #print(f"第一个decoder中的GCA输出形状: {final_output1.shape}") # torch.Size([2, 36, 32, 32])
        #开始进行残差连接
        # 将DenseNet的512通道输出降维到36通道（与rca_output1通道一致）
        adapted_densenet1 = self.channel_adapter1(densenet_input1)  # 形状：[2, 36, 32, 32]
        skip_connection1 = final_output1 + adapted_densenet1  # 残差相加（通道匹配）
        #print(f"第一个decoder中残差连接后形状: {skip_connection1.shape}")  # 输出：[2, 36, 32, 32]
        upsampled_output1 = self.transpose_conv1(skip_connection1)  # [2, 36, 64, 64]
        #print(f"第一个decoder添加转置卷积后形状: {upsampled_output1.shape}")


        #--------------------开始通过第二个decoder模块-----------------
        # 通过RCA模块
        #print("--------------开始通过第二个decoder模块-----------------")
        rca_output2 = self.rca_block2(upsampled_output1)
        #print("第二个decoder中通过RCA模块的输出形状",rca_output2.shape) #  torch.Size([2, 36, 64, 64])
        final_output2 = self.gca_block2(rca_output2)
        #print(f"第二个decoder中的GCA输出形状: {final_output2.shape}") #  torch.Size([2, 36, 64, 64])
        #开始进行残差连接
        adapted_densenet2 = self.channel_adapter2(densenet_input2)  # 
        skip_connection2 = final_output2 + adapted_densenet2  # 尺寸匹配后相加
        #print(f"第二个decoder残差连接后形状: {skip_connection2.shape}") # torch.Size([2, 36, 64, 64])
        upsampled_output2 = self.transpose_conv2(skip_connection2)   
        #print(f"第二个decoder添加转置卷积后形状: {upsampled_output2.shape}") # torch.Size([2, 36, 128, 128])
        

        #--------------------开始通过第三个decoder模块--------------------
        # 通过RCA模块
        #print("--------------开始通过第三个decoder模块-----------------")
        rca_output3 = self.rca_block2(upsampled_output2)
        #print("第三个decoder中通过RCA模块的输出形状",rca_output3.shape) #  torch.Size([2, 36, 128, 128])
        final_output3 = self.gca_block2(rca_output3)
        #print(f"第三个decoder中的GCA输出形状: {final_output3.shape}") #  torch.Size([2, 36, 128, 128])
        #开始进行残差连接
        adapted_densenet3 = self.channel_adapter3(densenet_input3)  # 
        skip_connection3 = final_output3 + adapted_densenet3  # 尺寸匹配后相加
        #print(f"第三个decoder残差连接后形状: {skip_connection3.shape}") # torch.Size([2, 36, 128, 128])
        upsampled_output3 = self.transpose_conv3(skip_connection3)   
        #print(f"第三个decoder添加转置卷积后形状: {upsampled_output3.shape}") # torch.Size([2, 36, 256, 256])

        #--------------------开始通过第四个decoder模块--------------------
        #print("--------------开始通过第四个decoder模块-----------------")
        # 这个就是把第三个decoder最后的输出和densenet的第一个block的输出进行拼接
        residual = self.channel_adapter4(densenet_input4)
        #print(f"第四个decoder中残差连接前形状: {residual.shape}")
        final = residual + upsampled_output3
        #print(f"第四个decoder的输入: {final.shape}")# torch.Size([2, 36, 256, 256])

        # 新增：应用两个 Conv3×3+BN 模块
        x = self.conv_bn(final)
        x = self.conv_bn(x)
        x = self.final_upsample(x)  # 256→512
        x = self.final_conv(x)
        #x = self.sigmoid(x)  # 输出范围 [0, 1]
        x = self.softmax(x)
        #print(f"最终输出形状: {x.shape}")  # 输出: torch.Size([2, 36, 512, 512])

        
        return x


# 示例验证
if __name__ == "__main__":
    # 模拟512×512的RGB输入（批次2）
    input_img = torch.randn(2, 3, 512, 512)
    # 初始化组合模型
    model = CombinedModel()
    # 前向传播
    output = model(input_img)
    # 打印各阶段形状
    print(f"输入形状: {input_img.shape}")
    print(f"最终输出形状: {output.shape}")