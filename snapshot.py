import time
import mss
import cv2
import numpy as np
from datetime import datetime
import onnxruntime

left = 850
top = 400
width = 224
dst_width = 1920
dst_height = 1080

src_width = 2560
src_height = 1440

scale = src_height / dst_height

dst_left = int(left * scale)
dst_top  = int(top * scale)
new_width = int(width * scale)


ort_session = onnxruntime.InferenceSession('./runs/PPLCNet_x1_5.onnx')  

start_time = time.time()
frame_count = 0

with mss.mss() as sct:
    while True:
        img = sct.grab({
            'left': dst_left, 'top': dst_top, 
            'width': new_width, 'height': new_width
        })
        img = np.array(img)
        img = cv2.resize(img, (width, width))[..., :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_in = (img.transpose((2, 0, 1)) / 255.0).astype(np.float32).reshape(1, 3, 224, 224)

        pred = ort_session.run(None, {'input': img_in})[0][0]

        pred_label = pred.argmax()
        print(f'[{frame_count}]', pred_label, f'{frame_count / (time.time() - start_time):.2f} FPS', pred)
        
        # cv2.imwrite(f'data/snapshot07252130/{i:07d}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # time.sleep(0.05)
        frame_count += 1
