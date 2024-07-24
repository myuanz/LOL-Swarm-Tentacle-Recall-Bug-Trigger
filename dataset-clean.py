# %%
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
# %%
train_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.RandomErasing(p=0.2),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = ImageDataset("./data/frames", "./labels.json", transform=eval_transform)

# %%
img_ts, label = dataset[80]
img = img_ts.numpy().transpose((1, 2, 0))
plt.imshow(img)
plt.title(label)
# %%
model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 classes
model.eval()

if os.path.exists('image_classifier.pth'):
    model.load_state_dict(torch.load('image_classifier.pth'))
# %%
imgs_ts = torch.stack([img for img, _ in dataset])
labels = torch.tensor([label for _, label in dataset])
# %%
preds = model(imgs_ts)
# %%
preds_label = preds.argmax(dim=1)
preds_label
# %%
confusion_matrix(labels, preds_label.numpy())
# %%
imgs_confusion_mtx = {}
for pred_label, true_label, img_ts in zip(preds_label, labels, imgs_ts):
    k = (pred_label.item(), true_label.item())
    
    if k not in imgs_confusion_mtx:
        imgs_confusion_mtx[k] = []
    imgs_confusion_mtx[k].append(img_ts)

# %%


imgs_confusion_mtx[(0, 1)]
# %%
from torchvision.utils import make_grid

to_show = imgs_confusion_mtx[(1, 0)][:80]

grid = make_grid(to_show, nrow=8, padding=2, normalize=True)

grid = grid.numpy().transpose((1, 2, 0))

# 显示图像
plt.figure(figsize=(15, 15))
plt.imshow(grid)
plt.axis('off')
plt.show()