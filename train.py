import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
import cv2

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, image_folder, labels_file, transform=None):
        self.transform = transform
        self.label_types = ["普通人物", "动作人物", "选择卡片", "其他"]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_types)}

        # 加载标签
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)

        # 初始化图像和标签列表
        self.images = []
        self.label_indices = []

        # 一次性加载所有图像到内存
        for img_name, label in self.labels.items():
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

    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
dataset = ImageDataset("./data/frames", "./labels.json", transform=None)
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

train_dataset = ImageDataset("./data/frames", "./labels.json", transform=train_transform)
test_dataset = ImageDataset("./data/frames", "./labels.json", transform=test_transform)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler)

# 定义模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置 TensorBoard 2024-0618-2305
writer = SummaryWriter(log_dir=f'runs/{datetime.now():%Y%m%d-%H%M}')

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

        if i % 10 == 9:
            writer.add_scalar('training loss', running_loss / 10, epoch * len(loader) + i)
            running_loss = 0.0

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
num_epochs = 20
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    test_loss, accuracy = test(model, test_loader, criterion, epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

writer.close()

# 保存模型
torch.save(model.state_dict(), 'image_classifier.pth')