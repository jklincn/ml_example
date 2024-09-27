import argparse
import os
import time

import torch
import torch_mlu
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os


default_num_epochs = 1
default_batch_size = 64
device = "mlu"

# Argument parser setup
parser = argparse.ArgumentParser(description="Distributed training script")
parser.add_argument(
    "--num_epochs",
    type=int,
    default=default_num_epochs,
    help="number of epochs to train",
)
parser.add_argument(
    "--batch_size", type=int, default=default_batch_size, help="batch size"
)
args = parser.parse_args()

# Use values from argparse or default
num_epochs = args.num_epochs
batch_size = args.batch_size


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("cncl", rank=rank, world_size=world_size)
    torch.mlu.set_device(rank)


def cleanup():
    # 销毁进程组
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(world_size)
    torch.mlu.manual_seed(world_size)

    # 2. 定义数据预处理和加载
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # VGG 网络需要输入大小为 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 下载 CIFAR-10 数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # 分布式数据采样器
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=train_sampler,
    )

    # 3. 加载预训练的 VGG19 模型并修改输出层
    model = models.vgg19(weights="VGG19_Weights.DEFAULT")
    model.classifier[6] = nn.Linear(4096, 10)  # 将最后一层的输出改为 10 类（CIFAR-10）
    model = model.to(device)

    # 使用 DistributedDataParallel 包装模型
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 开始训练
    model.train()
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # 确保每个 epoch 数据是随机的

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0 and rank == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Iteration {i}, Loss: {loss.item():.4f}",
                    flush=True,
                )

        if rank == 0:  # 只有 rank 0 的进程打印日志
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}"
            )

    cleanup()


if __name__ == "__main__":
    torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    models.vgg19(weights="VGG19_Weights.DEFAULT")

    world_size = torch.mlu.device_count()

    start_time = time.time()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")
