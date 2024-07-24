import json
import os
from collections import Counter
from datetime import datetime

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
from pathlib import Path

# 定义数据集类
class ImageDataset(Dataset):
    label_types = ["普通人物", "动作人物", "选择卡片", "其他"]
    
    def __init__(self, image_folder: str, labels: dict[str, str], transform=None, included_img_names: set[str]=set()):
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_types)}

        self.labels = labels

        # 初始化图像和标签列表
        self.images = []
        self.label_indices = []

        # 一次性加载所有图像到内存
        for img_name, label in self.labels.items():
            if included_img_names and img_name not in included_img_names:
                continue

            img_path = os.path.join(image_folder, img_name)
            image = cv2.imread(img_path)
            self.images.append(image)
            self.label_indices.append(self.label_to_idx[label])

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
    transforms.RandomRotation(10),
    transforms.RandomErasing(p=0.2),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

all_labels: dict[str, str] = json.load(open('./labels.json', 'r'))
train_img_names, test_img_names = train_test_split(list(all_labels.keys()), test_size=0.2)


class_counts = Counter(all_labels.values())
num_samples = len(all_labels)
class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
sample_weights = [class_weights[all_labels[n]] for n in train_img_names]

sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)

train_dataset = ImageDataset("./data/frames", all_labels, transform=train_transform, included_img_names=set(train_img_names))
test_dataset = ImageDataset("./data/frames", all_labels, transform=test_transform, included_img_names=set(test_img_names))


train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=256)

# 定义模型
model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if os.path.exists('image_classifier.pth'):
    model.load_state_dict(torch.load('image_classifier.pth'))

# 定义损失函数和优化器
criterion_weight = torch.Tensor([
    class_weights[cls] 
    for cls in ImageDataset.label_types
]).to(device)
print(f'{criterion_weight=} {class_weights=}')
criterion = nn.CrossEntropyLoss(weight=criterion_weight)
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
    
#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
#         return focal_loss.mean()

# criterion = FocalLoss()

optimizer = optim.SGD(model.parameters(), lr=0.0005)

log_dir = Path(f'runs/{datetime.now():%Y%m%d-%H%M}')
writer = SummaryWriter(log_dir=log_dir)

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

        writer.add_scalar('training loss', running_loss, epoch * len(loader) + i)
        running_loss = 0.0
    # print(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    writer.add_figure('training confusion matrix', fig, epoch)

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

    writer.add_scalar('test loss', test_loss, epoch)
    writer.add_scalar('test accuracy', accuracy, epoch)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    writer.add_figure('test confusion matrix', fig, epoch)

    return test_loss, accuracy

# 训练循环
num_epochs = 200
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    test_loss, accuracy = test(model, test_loader, criterion, epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    torch.save(model.state_dict(), 'image_classifier.pth')
writer.close()

# 保存模型
