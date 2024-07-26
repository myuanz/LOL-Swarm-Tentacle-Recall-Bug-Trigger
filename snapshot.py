import time
import mss
import cv2
import numpy as np

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

with mss.mss() as sct:
    i = 0
    while True:
        img = sct.grab({
            'left': dst_left, 'top': dst_top, 
            'width': new_width, 'height': new_width
        })
        img = np.array(img)
        img = cv2.resize(img, (width, width))
        
        cv2.imwrite(f'data/snapshot/{i:07d}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        time.sleep(1)
        i += 1
