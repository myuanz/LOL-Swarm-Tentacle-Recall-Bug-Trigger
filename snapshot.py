import time
import mss
import cv2
import numpy as np
import torch
import pplcnet
from torchvision import transforms
from datetime import datetime

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


model = pplcnet.PPLCNet_x1_0(num_classes=4)
model.load_state_dict(torch.load('PPLCNet1.0.pth', map_location='cpu'))
model.eval()

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])


with mss.mss() as sct:
    i = 0
    while True:
        img = sct.grab({
            'left': dst_left, 'top': dst_top, 
            'width': new_width, 'height': new_width
        })
        img = np.array(img)
        img = cv2.resize(img, (width, width))[..., :3]
        img_ts = eval_transform(img).unsqueeze(0)
        pred = model(img_ts)
        pred_label = pred.argmax(dim=1).item()
        print(datetime.now(), pred_label, pred)
        # cv2.imwrite(f'data/snapshot07252130/{i:07d}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        time.sleep(0.05)
        i += 1
