import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from beartype import beartype
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler

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

    def split(self, test_size=0.2, train_transform: Callable|None=None, test_transform: Callable|None=None, random_seed=42):
        train_idxs, test_idxs = train_test_split(
            range(len(self.images)),
            test_size=test_size, random_state=random_seed
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
            image = self.transform(image=image)['image']
        else:
            image = torch.from_numpy(np.array(image).transpose((2, 0, 1)))

        return image, label

    def get_class_weights(self) -> np.ndarray:
        class_weights = np.zeros(len(self.label_types))
        for label in self.label_indices:
            class_weights[label] += 1
        class_weights = 1 / class_weights
        class_weights /= class_weights.sum()
        return class_weights

    def get_weighted_sampler(self):
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in self.label_indices]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

if __name__ == '__main__':
    dataset = ImageDataset(Path('data/frames'))
    print(dataset.get_class_weights())
