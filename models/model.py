import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=37):
    # 加载预训练的ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # 替换最后一层全连接层，输出37类
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# 定义Label Smoothing损失函数
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        # 创建平滑标签
        smooth_target = torch.full_like(pred, self.smoothing / (n_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        # 计算KL散度
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        loss = -(smooth_target * log_prob).sum(dim=-1).mean()
        return loss
