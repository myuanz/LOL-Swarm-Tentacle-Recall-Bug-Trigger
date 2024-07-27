import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import tyro
from beartype import beartype
from serde import serde
import serde.json as sjson

@serde
@dataclass
class TrainConfig:
    model: Literal[
        'PPLCNet_x0_25', 'PPLCNet_x0_35', 'PPLCNet_x0_5', 'PPLCNet_x0_75', 
        'PPLCNet_x1_0', 'PPLCNet_x1_5', 'PPLCNet_x2_0', 'PPLCNet_x2_5',
        'resnet18'
    ] = 'PPLCNet_x1_5'

    optimizer: Literal['SGD', 'Adam', 'AdamW'] = 'Adam'
    lr: float = 0.0005
    batch_size: int = 256
    num_epochs: int = 200

    load_model: Path | None = None
    log_dir: Path = Path(f'runs/{datetime.now():%Y%m%d-%H%M%S}')
    last_epoch: int = 0
    random_seed: int = 42
    test_size: float = 0.2
    
    def save(self):
        self.log_dir.joinpath('config.json').write_text(sjson.to_json(self))


args = tyro.cli(TrainConfig)
if args.log_dir.exists():
    raise FileExistsError(f'{args.log_dir} already exists')
if args.load_model and not args.load_model.exists():
    raise FileNotFoundError(f'{args.load_model} does not exist')

args.log_dir.mkdir(parents=True, exist_ok=True)
args.save()

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from dataset import ImageDataset
import pplcnet

# 定义数据增强和归一化
train_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomRotation(10),
    transforms.RandomErasing(p=0.2),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])


total_dataset = ImageDataset(
    [Path("./data/frames"), Path('./data/2024-07-25-21-36-48/')], 
)
train_dataset, test_dataset = total_dataset.split(test_size=args.test_size, train_transform=train_transform, test_transform=test_transform)


train_loader = DataLoader(
    train_dataset, batch_size=256, sampler=train_dataset.get_weighted_sampler()
)
test_loader = DataLoader(test_dataset, batch_size=256)

# 定义模型
if args.model == 'resnet18':
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # 4 classes
else:
    model = getattr(pplcnet, args.model)(num_classes=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if args.load_model:
    model.load_state_dict(torch.load(args.load_model))

# 定义损失函数和优化器
criterion_weight = torch.Tensor(train_dataset.get_class_weights()).to(device)
print(f'{criterion_weight=}')

criterion = nn.CrossEntropyLoss(weight=criterion_weight)
optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)

writer = SummaryWriter(log_dir=args.log_dir)

# 训练函数
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        writer.add_scalar('loss/training loss', running_loss, epoch * len(loader) + i)
        running_loss = 0.0
    # print(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    writer.add_figure('cfmtx/training confusion matrix', fig, epoch)

# 测试函数
def test(model, loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)

    writer.add_scalar('loss/test loss', test_loss, epoch)
    writer.add_scalar('test accuracy', accuracy, epoch)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    writer.add_figure('cfmtx/test confusion matrix', fig, epoch)

    return test_loss, accuracy

# 训练循环
num_epochs = args.num_epochs
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    test_loss, accuracy = test(model, test_loader, criterion, epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    torch.save(model.state_dict(), args.log_dir / f"{args.model}.pth")
    args.last_epoch = epoch
writer.close()
args.save()

# 保存模型
