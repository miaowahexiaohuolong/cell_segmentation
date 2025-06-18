import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = torch.cat([x, out], dim=1)  # 密集连接，拼接输入
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate  # 每添加一层，通道数增加growth_rate
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DenseNetEncoder(nn.Module):
    def __init__(self):
        super(DenseNetEncoder, self).__init__()
        growth_rate = 32  # 生长率，与DenseLayer的卷积输出通道数一致
        # 定义四个 Dense Block 及后续操作（输入为3通道RGB图像）
        self.dense_block1 = DenseBlock(num_layers=4, in_channels=3, growth_rate=growth_rate)
        self.conv1 = nn.Conv2d(3 + 4 * growth_rate, 64, kernel_size=1)  # 调整通道到64
        self.pool1 = nn.MaxPool2d(2)  # 下采样（512→256）
        
        self.dense_block2 = DenseBlock(num_layers=4, in_channels=64, growth_rate=32)
        self.conv2 = nn.Conv2d(64 + 4 * 32, 128, kernel_size=1)  # 调整通道到128
        self.pool2 = nn.MaxPool2d(2)  # 下采样（256→128）
        
        self.dense_block3 = DenseBlock(num_layers=4, in_channels=128, growth_rate=32)
        self.conv3 = nn.Conv2d(128 + 4 * 32, 256, kernel_size=1)  # 调整通道到256
        self.pool3 = nn.MaxPool2d(2)  # 下采样（128→64）
        
        self.dense_block4 = DenseBlock(num_layers=4, in_channels=256, growth_rate=32)
        self.conv4 = nn.Conv2d(256 + 4 * 32, 512, kernel_size=1)  # 调整通道到512
        self.pool4 = nn.MaxPool2d(2)  # 下采样（64→32）

    def forward(self, x):
        features = []
        # Block 1 输出：[b, 64, 256, 256]
        x = self.dense_block1(x)
        x = self.conv1(x)
        x = self.pool1(x)
        features.append(x)
        
        # Block 2 输出：[b, 128, 128, 128]
        x = self.dense_block2(x)
        x = self.conv2(x)
        x = self.pool2(x)
        features.append(x)
        
        # Block 3 输出：[b, 256, 64, 64]
        x = self.dense_block3(x)
        x = self.conv3(x)
        x = self.pool3(x)
        features.append(x)
        
        # Block 4 输出：[b, 512, 32, 32]
        x = self.dense_block4(x)
        x = self.conv4(x)
        x = self.pool4(x)
        features.append(x)
        
        return features  # 返回张量列表

if __name__ == "__main__":
    model = DenseNetEncoder()
    input_tensor = torch.randn(1, 3, 512, 512)  # 输入形状：[b, 3, 512, 512]
    
    # 获取四个中间输出（列表）
    intermediate_outputs = model(input_tensor)
    
    # 打印每个中间输出的形状
    print("输入张量形状：", input_tensor.shape)  # 输出：torch.Size([1, 3, 512, 512])
    print("\n各中间输出形状：")
    for i, feat in enumerate(intermediate_outputs, 1):
        print(f"Block {i} 输出形状：{feat.shape}")

    # 示例输出：
    # 各中间输出形状：
    # Block 1 输出形状：torch.Size([1, 64, 256, 256])
    # Block 2 输出形状：torch.Size([1, 128, 128, 128])
    # Block 3 输出形状：torch.Size([1, 256, 64, 64])
    # Block 4 输出形状：torch.Size([1, 512, 32, 32])