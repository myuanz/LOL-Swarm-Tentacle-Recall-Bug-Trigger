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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from pathlib import Path
from tqdm import tqdm
from dataset import ImageDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
eval_transform = A.Compose([
    A.Normalize(mean=0, std=1),
    ToTensorV2(),
])

dataset = ImageDataset([
    Path("./data/2024-07-25-21-36-48/")
], transform=eval_transform)
train_dataset, test_dataset = dataset.split(test_size=0.2, train_transform=eval_transform, test_transform=eval_transform)
# %%
len(train_dataset), len(test_dataset), len(dataset)
# %%
Counter(dataset.all_labels.values())
# %%
img_ts, label = dataset[80]
img = img_ts.numpy().transpose((1, 2, 0))
plt.imshow(img)
plt.colorbar()
plt.title(label)
# %%
from pplcnet import PPLCNet_x1_5
from torchvision import models

# model = PPLCNet_x1_5(num_classes=4).to(device)
# model.load_state_dict(torch.load('runs/20240726-225616/PPLCNet_x1_5.pth'))
# model.load_state_dict(torch.load('runs/20240728-162337/PPLCNet_x1_5.pth'))

model = models.resnet18(num_classes=4).to(device)
model.load_state_dict(torch.load('runs/20240728-182521/resnet18.pth'))
model.eval()

# %%
imgs_ts = torch.stack([img for img, _ in dataset])
labels = torch.tensor([label for _, label in dataset])
# %%

dataloader = DataLoader(dataset, batch_size=32, pin_memory=True)
preds = []

tensor_in_cuda = next(iter(dataloader))[0].to(device)
with torch.no_grad():
    for imgs, _ in tqdm(dataloader):
        B = imgs.size(0)
        tensor_in_cuda[:B, ...].copy_(imgs)
        
        pred = model(tensor_in_cuda[:B]).cpu()
        preds.append(pred)
    preds = torch.cat(preds)
# %%
preds_label = preds.argmax(dim=1)
preds_label
# %%
confusion_matrix(labels, preds_label.numpy())
# %%
imgs_confusion_mtx = {}
for i, (pred_label, true_label, img_ts) in enumerate(zip(preds_label, labels, imgs_ts)):
    k = (pred_label.item(), true_label.item())
    
    if k not in imgs_confusion_mtx:
        imgs_confusion_mtx[k] = []
    imgs_confusion_mtx[k].append((img_ts, i))

# %%

len(imgs_confusion_mtx[(0, 1)])
# %%
from torchvision.utils import make_grid

to_show = [i[0] for i in imgs_confusion_mtx[(1, 0)][:40]]

grid = make_grid(to_show, nrow=4, padding=2, normalize=True)

grid = grid.numpy().transpose((1, 2, 0))

# 显示图像
plt.figure(figsize=(15, 15))
plt.imshow(grid)
plt.axis('off')
plt.show()
# %%
[dataset.image_names[i[1]] for i in imgs_confusion_mtx[(1, 0)]]
# %%
base_path = Path('data/snapshot')

for img_p in tqdm(base_path.glob('*.jpg')):
    img = cv2.imread(str(img_p))
    img_ts = eval_transform(img).unsqueeze(0)
    pred = model(img_ts)
    label = ['normal', 'action', 'card', 'other'][pred.argmax(dim=1).item()]
    
    plt.imshow(img)
    plt.title(f'{label} - {[f"{i:.2f}" for i in pred.tolist()[0]]}')
    plt.savefig(f'./data/snapshot/pred_{img_p.stem}.png')
    plt.close()
    # break
# %%
pred