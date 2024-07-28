import winsound
import time
import mss
import cv2
import numpy as np
from datetime import datetime
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

start_time = time.time()
frame_count = 0

exportor = BeepExportor()
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
        exportor.add_pred(pred=pred, t=t)
        pred_label = pred.argmax()
        print(f'[{frame_count}]', pred_label, f'beep: {exportor.pred_beep_time:.2f}s', f'{frame_count / (time.time() - start_time):.2f} FPS', pred)
        if exportor.pred_beep_time > 0 and 0 < exportor.pred_beep_time - t < 1:
            winsound.Beep(1000, int((exportor.pred_beep_time - t) * 1000))
        # cv2.imwrite(f'data/snapshot07252130/{i:07d}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # time.sleep(0.05)
        frame_count += 1
