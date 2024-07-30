import time
import winsound
from datetime import datetime
from threading import Thread
from collections import deque
import queue
import cv2
import mss
import numpy as np
import onnxruntime

from check_video import BeepExportor

top_ = 330/1080
left_ = 775/1920
width_ = 392/1920

curr_monitor_size = (2560, 1440)

top = int(top_ * curr_monitor_size[1])
left = int(left_ * curr_monitor_size[0])
width = int(width_ * curr_monitor_size[0])
print(top, left, width)
# ort_session = onnxruntime.InferenceSession('./runs/PPLCNet_x1_5.onnx')  
ort_session = onnxruntime.InferenceSession('./runs/resnet18.onnx')

beep_events: deque[tuple[float, int, int]] = deque() # (beep_time, freq, length)

def beep_thread():
    while True:
        if len(beep_events) == 0:
            time.sleep(0.01)
            continue
        beep_time, freq, length = beep_events[0]
        dt = time.time() - beep_time

        if dt < 0:
            continue

        if dt > 1:
            beep_events.popleft()
            continue

        beep_events.popleft()
        winsound.Beep(freq, length)

def run_snapshot(exportor: BeepExportor):
    start_time = time.time()
    frame_count = 0
    last_put_beep = 0
    last_event = None
    
    with mss.mss() as sct:
        while True:
            t = time.time()
            img = sct.grab({
                'left': left, 'top': top, 
                'width': width, 'height': width
            })
            img = np.array(img)
            img = cv2.resize(img, (224, 224))[..., :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_in = (img.transpose((2, 0, 1)) / 255.0).astype(np.float32).reshape(1, 3, 224, 224)

            pred = ort_session.run(None, {'input': img_in})[0][0]
            curr_event = exportor.add_pred(pred=pred, t=t)
            
            pred_label = pred.argmax()
            pred_act_time = exportor.pred_beep_time or 0
            
            if pred_label not in {0, 3}:
                print(f'[{t-start_time:.3f}]', pred_label, f'beep: {pred_act_time-start_time:.3f}s', f'{frame_count / (time.time() - start_time):.2f} FPS. beep queue: {len(beep_events)}', pred, end='\r', flush=True)
            else:
                print(f'[{t-start_time:.3f}]', end='\r', flush=True)
            if last_event is None or curr_event.label != last_event.label:
                print()
            last_event = curr_event
            
            if pred_act_time and pred_act_time > 0 and last_put_beep != pred_act_time:
                dt = pred_act_time - t
                if 1.5 > dt > 0:
                    beep_events.append((pred_act_time - 1, int(440/1.414), int(dt * 600)))
                    last_put_beep = pred_act_time
                    print(f'beep: {pred_act_time-start_time:.3f}s')

            # cv2.imwrite(f'data/snapshot07252130/{i:07d}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # time.sleep(0.05)
            frame_count += 1



def main():
    # exportor = BeepExportor(on_beep_act=lambda t: beep_events.append((time.time(), 440, 200)))
    exportor = BeepExportor()

    Thread(target=beep_thread, daemon=True).start()
    try:
        run_snapshot(exportor)
    except KeyboardInterrupt:
        import pickle
        with open(f'data/{datetime.now().strftime("%Y%m%d%H%M%S")}.pkl', 'wb') as f:
            pickle.dump(exportor, f)

if __name__ == '__main__':
    main()
