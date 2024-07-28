# %%
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

import cv2
import numpy as np
import onnxruntime
import orjson
import tyro
from moviepy.editor import VideoFileClip
from serde import serde, to_dict
from serde.json import to_json
from tqdm import tqdm

from image_utils import crop_image


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
    def from_pred(pred: np.ndarray, t: float=-1) -> 'PredWithTime':
        if t == -1:
            t = time.time()
        return PredWithTime(pred=pred, label=pred.argmax().item(), time=t)
    
    def __repr__(self):
        return f'Pred({self.label} | {", ".join(f"{p:.2f}" for p in self.pred)} @ {self.time:.3f}s)'
@dataclass
class PredWithIdx:
    pred: PredWithTime
    idx: int

NORMAL_EVENT: int = 0
ACT_EVENT   : int = 1
CARD_EVENT  : int = 2
OTHER_EVENT : int = 3
class BeepExportor(Exportor):
    def __init__(self):
        self.preds: list[PredWithTime] = []
        self.label_index: dict[int, list[int]] = {
            ACT_EVENT: [], NORMAL_EVENT: [], 
            CARD_EVENT: [], OTHER_EVENT: [],
        }
        self.tgt_events: list[PredWithTime] = [] # 仅包含 ACT_EVENT 和 CARD_EVENT, 且只有一簇的首个
        self.last_event = PredWithTime(np.zeros(4), -1, 0)
        self.pred_beep_time = 0
        self.i = 0
        
    def seek(self, label: int, i: int=-1) -> PredWithIdx:
        print(self.label_index, label, i)
        idx = self.label_index[label][i]
        return PredWithIdx(
            pred=self.tgt_events[idx],
            idx=idx
        )
    
    def count(self, label: int) -> int:
        return len(self.label_index[label])

    def add_frame(self, frame: np.ndarray, t: float) -> None:
        pass
    def add_pred(self, pred: np.ndarray, t: float) -> None:
        curr_pred = PredWithTime.from_pred(pred, t=t)
        print(curr_pred, self.pred_beep_time, self.tgt_events)
        if curr_pred.label != self.last_event.label:
            # 当前事件类型与上一个不同才有用, 其他都是平凡的
            if curr_pred.label == ACT_EVENT:
                if self.count(ACT_EVENT) == 0 or curr_pred.time - self.seek(ACT_EVENT, -1).pred.time > 1:
                    # 间隔大于 1s, 可以认为是新的一簇
                    self.label_index[curr_pred.label].append(len(self.tgt_events))
                    self.tgt_events.append(curr_pred)
            elif curr_pred.label == CARD_EVENT:
                if self.count(CARD_EVENT) == 0 or curr_pred.time - self.seek(CARD_EVENT, -1).pred.time > 0.1:
                    # 间隔大于 0.1s, 可以认为是新的一簇
                    self.label_index[curr_pred.label].append(len(self.tgt_events))
                    self.tgt_events.append(curr_pred)
            
            if (
                curr_pred.label != ACT_EVENT 
            ) and self.label_index[ACT_EVENT].__len__() >= 2:
                # 当前不是 ACT_EVENT, 但 ACT_EVENT 有两个以上, 可以用来计算下次 act 在何时
                self.pred_beep_time = self.calc_beep_time()
                print(f'{self.pred_beep_time=}')
                
        if self.pred_beep_time > 0 and self.pred_beep_time - curr_pred.time < 0.5:
            # 距离预测的act时间小于0.5s, 则发出提示音
            print(f'将在 {self.pred_beep_time} 捶地')
            # import winsound
            # winsound.Beep(1000, int((curr_pred_beep_time - curr_pred.time) * 1000))
        
        self.preds.append(curr_pred)
        # if curr_pred.label in {ACT_EVENT, CARD_EVENT}:
        #     self.label_index[curr_pred.label].append(len(self.tgt_events))
            # self.tgt_events.append(curr_pred)

        self.i += 1
        self.last_event = curr_pred
    def calc_beep_time(self) -> float:
        assert len(self.label_index[ACT_EVENT]) >= 2
        start_act = self.seek(ACT_EVENT, -2)
        end_act = self.seek(ACT_EVENT, -1)

        duration = end_act.pred.time - start_act.pred.time
        leave_idxs = list(range(start_act.idx, end_act.idx))
        
        for curr_idx, next_idx in zip(leave_idxs, leave_idxs[1:]):
            curr_event = self.tgt_events[curr_idx]
            next_event = self.tgt_events[next_idx]
            
            if curr_event.label == CARD_EVENT:
                duration -= next_event.time - curr_event.time

        pred_beep_time = end_act.pred.time + duration
        leave_idxs = list(range(end_act.idx, len(self.tgt_events)))
        for curr_idx, next_idx in zip(leave_idxs, leave_idxs[1:]):
            curr_event = self.tgt_events[curr_idx]
            next_event = self.tgt_events[next_idx]
            
            if curr_event.label == CARD_EVENT:
                pred_beep_time += next_event.time - curr_event.time
        return pred_beep_time
        
    def release(self) -> None:
        pass

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
    export_to: Literal['video', 'image', 'beep'] = 'video'
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

    # exportor: Exportor = VideoExportor(export_path, 512, 512) if args.export_to == 'video' else ImageExportor(export_path)
    match args.export_to:
        case 'video':
            exportor: Exportor = VideoExportor(export_path, 512, 512)
        case 'image':
            exportor: Exportor = ImageExportor(export_path)
        case 'beep':
            exportor: Exportor = BeepExportor()
        case _:
            raise ValueError(f'unknown export_to: {args.export_to}')

    time_list = np.linspace(0, video_clip.duration, int(video_clip.duration * video_clip.fps / args.skip))
    for t in tqdm(time_list):
        if t > 15: break
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
    # res = tyro.cli(RunConfig)
    res = RunConfig(
        video_path=Path('./data/2024-07-25-21-36-48.mp4'),
        export_to='beep',
        skip=1,
        model_path=Path('./runs/resnet18.onnx'),
    )
    main(res)
