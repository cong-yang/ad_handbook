import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import Tensor
from torch.quantization import DeQuantStub
from torchvision.datasets import CIFAR10
from torchvision.models.mobilenetv2 import MobileNetV2
from torch.utils import data
from typing import Optional, Callable, List, Tuple

# 导入地平线量化相关依赖
from horizon_plugin_pytorch.functional import rgb2centered_yuv
import torch.quantization
from horizon_plugin_pytorch.march import March, set_march
from horizon_plugin_pytorch.quantization import (
    QuantStub,
    convert_fx,
    prepare_qat_fx,
    set_fake_quantize,
    FakeQuantState,
    check_model,
    compile_model,
    perf_model,
    visualize_model,
)
from horizon_plugin_pytorch.quantization.qconfig import (
    default_calib_8bit_fake_quant_qconfig,
    default_qat_8bit_fake_quant_qconfig,
    default_qat_8bit_weight_32bit_out_fake_quant_qconfig,
    default_calib_8bit_weight_32bit_out_fake_quant_qconfig,
)

# 配置log输出格式
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class AverageMeter(object):
    """计算并保存当前值与平均值"""

    def __init__(self, name: str, fmt=":f"):
        self.name = name     # 指标名称，例如 "Acc@1"
        self.fmt = fmt       # 打印格式
        self.reset()         # 初始化各项值

    def reset(self):
        # 重置所有统计值
        self.val = 0         # 当前值
        self.avg = 0         # 平均值
        self.sum = 0         # 总和
        self.count = 0       # 累计数量

    def update(self, val, n=1):
        # 更新统计值，val 是当前值，n 是样本数
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # 返回格式化的字符串
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    
def accuracy(output: Tensor, target: Tensor, topk=(1,)) -> List[Tensor]:
    """计算 top-k 准确率，比如 top-1 和 top-5"""

    with torch.no_grad():
        maxk = max(topk)  # 取出最大的 k
        batch_size = target.size(0)

        # 从输出中取出前 k 个预测结果
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # 转置方便后续比较
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 判断是否预测正确

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()  # 前 k 个中正确的数量
            res.append(correct_k.mul_(100.0 / batch_size))  # 转换成百分比准确率
        return res

def evaluate(
    model: nn.Module, data_loader: data.DataLoader, device: torch.device
) -> Tuple[AverageMeter, AverageMeter]:
    
    # 初始化统计器
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # 不进行梯度计算，提高效率
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)  # 数据移到设备上
            output = model(image)  # 模型推理
            output = output.view(-1, 10)  # 对输出 reshape 成 [batch, num_classes]
            acc1, acc5 = accuracy(output, target, topk=(1, 5))  # 计算准确率
            top1.update(acc1, image.size(0))  # 更新 top1 准确率统计器
            top5.update(acc5, image.size(0))  # 更新 top5 准确率统计器
            print(".", end="", flush=True)  # 输出进度点
        print()  # 换行

    return top1, top5  # 返回统计结果

def train_one_epoch(
    model: nn.Module,
    criterion: Callable,  # 损失函数，如 nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer,  # 优化器，如 Adam、SGD
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],  # 学习率调度器，可为 None
    data_loader: data.DataLoader,  # 训练数据
    device: torch.device,  # 使用的设备，如 cuda 或 cpu
) -> None:

    # 初始化统计器
    top1 = AverageMeter("Acc@1", ":6.3f")
    top5 = AverageMeter("Acc@5", ":6.3f")
    avgloss = AverageMeter("Loss", ":1.5f")

    model.to(device)  # 模型转到设备上

    # 遍历每个 batch
    for image, target in data_loader:
        image, target = image.to(device), target.to(device)  # 数据转设备
        output = model(image)  # 正向传播
        output = output.view(-1, 10)  # reshape 输出
        loss = criterion(output, target)  # 计算损失

        optimizer.zero_grad()  # 清除上一步残余梯度
        loss.backward()        # 反向传播计算梯度
        optimizer.step()       # 更新权重

        if scheduler is not None:
            scheduler.step()   # 学习率调度（如果有）

        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # 计算准确率
        top1.update(acc1, image.size(0))  # 更新 top1 准确率
        top5.update(acc5, image.size(0))  # 更新 top5 准确率
        avgloss.update(loss, image.size(0))  # 更新损失

        print(".", end="", flush=True)  # 输出进度
    print()  # 换行

    # 打印训练结果汇总
    print(
        "Full cifar-10 train set: Loss {:.3f} Acc@1"
        " {:.3f} Acc@5 {:.3f}".format(avgloss.avg, top1.avg, top5.avg)
    )

def prepare_data_loaders(
    data_path: str, train_batch_size: int, eval_batch_size: int
) -> Tuple[data.DataLoader, data.DataLoader]:
    normalize = transforms.Normalize(mean=0.0, std=128.0)

    def collate_fn(batch):
        batched_img = torch.stack(
            [
                torch.from_numpy(np.array(example[0], np.uint8, copy=True))
                for example in batch
            ]
        ).permute(0, 3, 1, 2)
        batched_target = torch.tensor([example[1] for example in batch])

        batched_img = rgb2centered_yuv(batched_img)
        batched_img = normalize(batched_img.float())

        return batched_img, batched_target

    train_dataset = CIFAR10(
        data_path,
        True,
        transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
            ]
        ),
        download=True,
    )

    eval_dataset = CIFAR10(
        data_path,
        False,
        download=True,
    )

    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=data.RandomSampler(train_dataset),
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    eval_data_loader = data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        sampler=data.SequentialSampler(eval_dataset),
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_data_loader, eval_data_loader