# 数据集路径
DATA_ROOT = "D:/ltf/五种白细胞分割-cls5-2.8k"  # 替换为实际路径

# 训练参数
BATCHSIZE_PER_CARD = 2       # 单卡批次大小（根据显存调整）
TOTAL_EPOCH = 100            # 总训练轮次
WEIGHTS_DIR = "./weights"    # 模型权重保存目录