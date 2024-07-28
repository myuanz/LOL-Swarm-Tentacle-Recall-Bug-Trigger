import inspect
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

import cv2
import numpy as np
import onnxruntime
import orjson
import tyro
from moviepy.editor import VideoFileClip
from serde import serde, to_dict
from serde.json import from_json, to_json
from tqdm import tqdm

from image_utils import crop_image

# left = 850
# top = 400
# width = 224
# dst_width = 1920
# dst_height = 1080

def build_preview(img: np.ndarray, pred: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (512, 512))
    pred_label = pred.argmax().item()
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
    def add_frame(self, frame: np.ndarray, t: float) -> None:
        ...
    @abstractmethod
    def add_pred(self, pred: np.ndarray, t: float) -> None:
        ...

    def add_record(self, frame: np.ndarray, pred: np.ndarray, t: float, preview=False) -> None:
        self.add_frame(frame, t)
        self.add_pred(pred, t)
        if preview:
            preview_img = build_preview(frame, pred)
            cv2.imwrite(str(self.base_path / f'preview_{t:.3f}.jpg'), preview_img)
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

        self.video_writer = cv2.VideoWriter(
            str(base_path / 'video.mp4'), 
            cv2.VideoWriter.fourcc(*'mp4v'), 
            10, (width, height)
        )
    
    def add_frame(self, frame: np.ndarray, t: float) -> None:
        self.video_writer.write(frame)
        
    def add_pred(self, pred: np.ndarray, t: float) -> None:
        self.preds.append(pred.reshape(-1))

    def release(self) -> None:
        self.video_writer.release()
        np.save(self.base_path / 'preds.npy', np.array(self.preds))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

@dataclass
class PredWithTime:
    pred: np.ndarray
    label: int
    time: float # seconds
    
    @staticmethod
    def from_pred(pred: np.ndarray, t: float=-1) -> Self:
        if t == -1:
            t = time.time()
        return PredWithTime(pred=pred, label=pred.argmax().item(), time=t)
    

class BeepExportor(Exportor):
    def __init__(self):
        self.preds: list[PredWithTime] = []
        self.label_index = {
            0: [], 1: [], 2: [], 3: []
        }
        self.i = 0
        
    def add_frame(self, frame: np.ndarray) -> None:
        pass
    def add_pred(self, pred: np.ndarray) -> None:
        self.preds.append(PredWithTime.from_pred(pred))
        curr_pred = self.preds[-1]
        self.label_index[curr_pred.label].append(curr_pred)
        
        
    def release(self) -> None:
        pass
    

class ImageExportor(Exportor):
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.preds = []
        self.i = 0

    def add_frame(self, frame: np.ndarray, t: float) -> None:
        cv2.imwrite(str(self.base_path / f'{t:.3f}.jpg'), frame)
        self.i += 1

    def add_pred(self, pred: np.ndarray, t: float) -> None:
        self.preds.append(pred.reshape(-1))

    def release(self) -> None:
        np.save(self.base_path / 'preds.npy', np.array(self.preds))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

@serde
@dataclass
class RunConfig:
    video_path: tyro.conf.Positional[Path]
    dst: Path|None = None
    export_to: Literal['video', 'image'] = 'video'
    skip: int = 10
    model_path: Path = Path('./runs/PPLCNet_x1_5.onnx')
    enable_predict: bool = True
    top: int|float = 330/1080
    left: int|float = 775/1920
    width: int|float = 392/1920
    

def main(args: RunConfig):
    print(f'use {args.video_path}...')

    ort_session = onnxruntime.InferenceSession(args.model_path) if args.enable_predict else None
    
    video_clip = VideoFileClip(str(args.video_path), audio=False)
    print(f'video duration: {video_clip.duration}s, fps: {video_clip.fps}')

    if args.dst is not None:
        export_path = args.dst
    else:
        export_path = args.video_path.with_name(args.video_path.stem)
    if export_path.is_file():
        raise FileExistsError(f'{export_path} already exists')
    export_path.mkdir(parents=True, exist_ok=True)

    exportor: Exportor = VideoExportor(export_path, 512, 512) if args.export_to == 'video' else ImageExportor(export_path)

    time_list = np.linspace(0, video_clip.duration, int(video_clip.duration * video_clip.fps / args.skip))
    for t in tqdm(time_list):
        frame = video_clip.get_frame(t)
        img = crop_image(frame, args.left, args.top, args.width)

        img = cv2.resize(img, (224, 224))[..., :3]
        img_in = (img.transpose((2, 0, 1)) / 255.0).astype(np.float32).reshape(
            1, 3, 224, 224
        )
        if ort_session:
            pred = ort_session.run(None, {'input': img_in})[0][0]
        else:
            pred = np.zeros(4)

        exportor.add_record(img, pred, t, preview=False)

    exportor.release()
    video_clip.close()
    open(export_path / 'config.json', 'w').write(
        to_json(args, option=orjson.OPT_INDENT_2)
    )


if __name__ == "__main__":
    # tyro.cli(main)
    res = tyro.cli(RunConfig)
    main(res)
