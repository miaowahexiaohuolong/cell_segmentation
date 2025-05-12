import torch
import torch.nn as nn


class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        # 显式获取模型设备（通过第一个参数的device属性）
        self.device = next(net.parameters()).device if next(net.parameters()).is_cuda else torch.device("cpu")
        self.net = net.to(self.device)  # 确保模型在正确设备（冗余，传入的net应已处理）
        self.loss = loss.to(self.device)  # 损失函数同步到模型设备
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.old_lr = lr

        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        # 输入数据同步到模型设备（使用保存的self.device）
        self.img = img_batch.to(self.device)
        if mask_batch is not None:
            self.mask = mask_batch.to(self.device)
        self.img_id = img_id

    def optimize(self):
        self.optimizer.zero_grad()
        pred = self.net(self.img)  # 模型前向传播（输入已同步到设备）
        loss = self.loss(pred, self.mask)  # 损失计算（设备一致）
        loss.backward()  # 反向传播
        self.optimizer.step()  # 参数更新
        return loss, pred

    def save(self, path):
        # 多卡训练时保存module的state_dict（避免DataParallel包装问题）
        if isinstance(self.net, nn.DataParallel):
            torch.save(self.net.module.state_dict(), path)
        else:
            torch.save(self.net.state_dict(), path)

    def load(self, path):
        # 多卡训练时加载module的state_dict
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.old_lr = new_lr