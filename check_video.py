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
from moviepy.video.fx.crop import crop

from image_utils import crop_image, BBox, size_to_bbox, Pair

DEBUG = False

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
class PredPeriod:
    begin: PredWithTime
    end: PredWithTime
    idx: int

    @property
    def period(self) -> float:
        return self.end.time - self.begin.time

    @property
    def begin_time(self) -> float:
        return self.begin.time
    
    @property
    def end_time(self) -> float:
        return self.end.time

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
        self.all_period: list[PredPeriod] = []
        self.last_event = PredWithTime(np.zeros(4), -1, 0)
        self.pred_beep_time = 0
        self.i = 0
        
    def seek(self, label: int, i: int=-1) -> PredPeriod:
        idx = self.label_index[label][i]
        return self.all_period[idx]
    
    def count(self, label: int) -> int:
        return len(self.label_index[label])

    def add_frame(self, frame: np.ndarray, t: float) -> None:
        pass
    def add_pred(self, pred: np.ndarray, t: float) -> None:
        curr_pred = PredWithTime.from_pred(pred, t=t)
        if DEBUG:
            print(curr_pred, self.pred_beep_time, self.all_period[-2:])
        
        if curr_pred.label == self.last_event.label and self.all_period[-1].begin.label == curr_pred.label:
            self.all_period[-1].end = curr_pred
        else:
            thr = {
                ACT_EVENT: 0.5, CARD_EVENT: 0.1, NORMAL_EVENT: 0.1, OTHER_EVENT: 0.1
            }[curr_pred.label]

            if (
                self.count(curr_pred.label) > 0
                and (last_same_label_event := self.seek(curr_pred.label, -1))
                and curr_pred.time - last_same_label_event.end_time < thr
            ):
                last_same_label_event.end = curr_pred
            else:
                self.label_index[curr_pred.label].append(len(self.all_period))

                self.all_period.append(PredPeriod(
                    begin=curr_pred, end=curr_pred,
                    idx=len(self.all_period),
                ))

        if curr_pred.label != ACT_EVENT and self.label_index[ACT_EVENT].__len__() >= 2:
            # 当前不是 ACT_EVENT, 但 ACT_EVENT 有两个以上, 可以用来计算下次 act 在何时
            self.pred_beep_time = self.calc_beep_time()
            # print(f'new {self.pred_beep_time=}')

        # if self.pred_beep_time > 0 and self.pred_beep_time - curr_pred.time < 0.5:
        #     # 距离预测的act时间小于0.5s, 则发出提示音
        #     print(f'将在 {self.pred_beep_time} 捶地')
            # import winsound
            # winsound.Beep(1000, int((curr_pred_beep_time - curr_pred.time) * 1000))
        
        self.preds.append(curr_pred)
        self.i += 1
        self.last_event = curr_pred

    def find_last_two_period_with_threshold(self, label: int, length_thr: float) -> Pair[PredPeriod] | None:
        if self.count(label) < 2:
            return None
        periods: list[PredPeriod] = []
        for i in range(-1, -len(self.label_index[label])-1, -1):
            if len(periods) == 2:
                break
            period = self.seek(label, i)
            if period.period > length_thr:
                periods.append(period)
        if len(periods) < 2:
            return None
        return Pair(periods[0], periods[1])

    def calc_beep_time(self) -> float | None:        
        last_periods = self.find_last_two_period_with_threshold(ACT_EVENT, length_thr=1/30)
        # print(f'{last_periods=}')
        if last_periods is None:
            return None
        else:
            second_act, first_act = last_periods.first, last_periods.second

        duration = second_act.begin_time - first_act.begin_time
        leave_idxs = list(range(first_act.idx, second_act.idx))
        
        for idx in leave_idxs:
            p = self.all_period[idx]
            if p.begin.label != CARD_EVENT:
                continue
            
            duration -= p.period

        pred_beep_time = second_act.begin_time + duration
        leave_idxs = list(range(second_act.idx + 1, len(self.all_period)))
        for idx in leave_idxs:
            p = self.all_period[idx]
            if p.begin.label != CARD_EVENT:
                continue
            pred_beep_time += p.period            

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
        cv2.imwrite(str(self.base_path / f'{t:08.3f}.jpg'), frame)
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
    box = size_to_bbox(
        video_clip.size[0], video_clip.size[1],
        args.left, args.top, args.width, args.width
    )
    video_clip = crop(
        video_clip, 
        x1=box.x, y1=box.y, width=box.w, height=box.h
    )

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
    time_list = [
        i for i in time_list if (i > 30 and i < 90) 
    ]
    if not DEBUG:
        time_list = tqdm(time_list)
    for t in (time_list):
        frame = video_clip.get_frame(t)
        # img = crop_image(frame, args.left, args.top, args.width)
        img = frame
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
    if 'get_ipython' in locals():
        res = RunConfig(
            video_path=Path(r'./data/2024-07-28-23-09-59.mp4'),
            export_to='beep',
            skip=1,
            model_path=Path('./runs/resnet18.onnx'),
        )
        DEBUG = True
    else:
        res = tyro.cli(RunConfig)
    main(res)

# %%
