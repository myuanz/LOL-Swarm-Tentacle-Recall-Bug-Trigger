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

import pplcnet


@beartype
class ImageDataset(Dataset):
    label_types = ["普通人物", "动作人物", "选择卡片", "其他"]
    
    def __init__(
        self, image_folder: list[Path] | Path, transform: Callable|None=None, included_img_names: set[str]=set(), *, 
        split_images: list[np.ndarray]=[], split_labels: list[int]=[], split_image_names: list[str]=[]
    ):
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_types)}
        self.included_img_names = included_img_names

        self.images: list[np.ndarray] = []
        self.label_indices: list[int] = []
        self.image_names: list[str] = []
        self.all_labels: dict[str, str] = {}

        if isinstance(image_folder, Path):
            self.image_folder = [image_folder]
        else:
            self.image_folder = image_folder

        if split_images and split_labels and split_image_names:
            self.images = split_images
            self.label_indices = split_labels
            self.image_names = split_image_names
        else:
            for folder in self.image_folder:
                self.load_folder(folder)

        self.all_labels = {img_name: self.label_types[label] for img_name, label in zip(self.image_names, self.label_indices)}

    def load_folder(self, image_folder: Path):
        curr_labels = json.load(open(image_folder / 'labels.json', 'r'))
        
        for img_name, label in curr_labels.items():
            full_name = f'{image_folder.name}/{img_name}'
            if self.included_img_names and full_name not in self.included_img_names:
                continue
            img_path = image_folder / img_name
            image = cv2.imread(str(img_path))
            self.images.append(image)
            self.label_indices.append(self.label_to_idx[label])
            self.image_names.append(full_name)

    def split(self, test_size=0.2, train_transform: Callable|None=None, test_transform: Callable|None=None):
        train_idxs, test_idxs = train_test_split(
            range(len(self.images)),
            test_size=test_size, random_state=args.random_seed
        )

        train_dataset = ImageDataset(
            self.image_folder, transform=train_transform, 
            split_images=[self.images[i] for i in train_idxs],
            split_labels=[self.label_indices[i] for i in train_idxs],
            split_image_names=[self.image_names[i] for i in train_idxs],
        )
        test_dataset = ImageDataset(
            self.image_folder, transform=test_transform, 
            split_images=[self.images[i] for i in test_idxs],
            split_labels=[self.label_indices[i] for i in test_idxs],
            split_image_names=[self.image_names[i] for i in test_idxs],
        )
        return train_dataset, test_dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.label_indices[idx]

        if self.transform:
            image = self.transform(image)
        else:
            # 如果没有提供 transform，至少将图像转换为 tensor
            image = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0

        return image, label

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

all_labels = total_dataset.all_labels

class_counts = Counter(all_labels.values())
num_samples = len(all_labels)
class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
sample_weights = [class_weights[all_labels[n]] for n in train_dataset.image_names]

sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)

train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
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
criterion_weight = torch.Tensor([
    class_weights[cls] 
    for cls in ImageDataset.label_types
]).to(device)
print(f'{criterion_weight=} {class_weights=}')

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
