import time
import mss
import cv2
import numpy as np
from datetime import datetime
import tyro
from pathlib import Path
from tqdm import tqdm
from typing import Literal, Protocol, Self
from abc import ABC, abstractmethod

left = 850
top = 400
width = 224
dst_width = 1920
dst_height = 1080

def build_preview(img: np.ndarray, pred: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (512, 512))
    pred_label = pred.argmax(dim=1).item()
    if pred_label == 1:
        mask = np.zeros_like(img)
        mask[:, :, 1] = 255
        img = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
    cv2.putText(img, f"{pred_label} | {', '.join(f'{p:.2f}' for p in pred[0])}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img

class Exportor(ABC):
    base_path: Path
    i: int
    

    @abstractmethod
    def add_frame(self, frame: np.ndarray) -> None:
        ...
    @abstractmethod
    def add_pred(self, pred: np.ndarray) -> None:
        ...

    def add_record(self, frame: np.ndarray, pred: np.ndarray, preview=False) -> None:
        self.add_frame(frame)
        self.add_pred(pred)
        if preview:
            preview_img = build_preview(frame, pred)
            cv2.imwrite(str(self.base_path / f'preview_{self.i:07d}.jpg'), preview_img)
    @abstractmethod
    def release(self) -> None:
        ...
    @abstractmethod
    def __enter__(self: Self) -> Self:
        ...
    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...
    
class VideoExportor(Exportor):
    def __init__(self, base_path: Path, width: int, height: int):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.preds = []

        self.video_writer = cv2.VideoWriter(str(base_path / 'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    
    def add_frame(self, frame: np.ndarray) -> None:
        self.video_writer.write(frame)
        
    def add_pred(self, pred: np.ndarray) -> None:
        self.preds.append(pred.reshape(-1))

    def release(self) -> None:
        self.video_writer.release()
        np.save(self.base_path / 'preds.npy', np.array(self.preds))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        
class ImageExportor(Exportor):
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.preds = []
        self.i = 0
    
    def add_frame(self, frame: np.ndarray) -> None:
        cv2.imwrite(str(self.base_path / f'{self.i:07d}.jpg'), frame)
        self.i += 1
        
    def add_pred(self, pred: np.ndarray) -> None:
        self.preds.append(pred.reshape(-1))

    def release(self) -> None:
        np.save(self.base_path / 'preds.npy', np.array(self.preds))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def main(video_path: Path, /, dst: Path|None = None, export_to: Literal['video', 'image'] = 'mp4', skip: int = 10):
    print(f'use {video_path}...')
    import torch
    import pplcnet
    from torchvision import transforms

    model = pplcnet.PPLCNet_x1_0(num_classes=4)
    model.load_state_dict(torch.load('PPLCNet1.0.pth', map_location='cpu'))
    model.eval()

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    video_cap = cv2.VideoCapture(video_path)
    if dst is not None:
        export_path = dst
    else:
        export_path = video_path.with_name(video_path.stem)
    if export_path.is_file():
        raise FileExistsError(f'{export_path} already exists')
    export_path.mkdir(parents=True, exist_ok=True)

    exportor: Exportor = VideoExportor(export_path, 512, 512) if export_to == 'video' else ImageExportor(export_path)

    bar = tqdm(total=int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    i = 0
    while True:
        i+=1
        bar.update(1)


        ret, img = video_cap.read()
        if not ret:
            break
        if i % skip != 0:
            continue


        img = img[top:top+width, left:left+width]
        img_ts = eval_transform(img).unsqueeze(0)
        pred = model(img_ts).detach().numpy()
        exportor.add_record(img, pred, preview=False)

    exportor.release()


if __name__ == "__main__":
    # Generate a CLI and call `main` with its two arguments: `foo` and `bar`.
    tyro.cli(main)