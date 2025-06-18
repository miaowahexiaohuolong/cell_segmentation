import os

# 数据集路径
DATA_ROOT = "D:/ltf/Training_R2_Jan2015"  # 替换为实际路径

# 训练参数
BATCHSIZE_PER_CARD = 2       # 单卡批次大小（根据显存调整）
TOTAL_EPOCH = 50         # 总训练轮次
WEIGHTS_DIR = "./weights"    # 模型权重保存目录
NUM_CLASSES = 1
BATCH_SIZE = 2
IMG_SIZE = (512, 512)
NUM_WORKERS = 12

# 获取所有帧 ID
frame_ids = sorted([
    item[:-6] for item in os.listdir(DATA_ROOT)
    if item.startswith("frame") and item.endswith("_stack")
])

# 划分训练集和测试集
train_ratio = 0.8
train_size = int(len(frame_ids) * train_ratio)
TRAIN_FRAME_IDS = frame_ids[:train_size]
TEST_FRAME_IDS = frame_ids[train_size:]