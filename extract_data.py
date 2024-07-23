# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import umap
import pickle

# %%

video_path = Path('data/2024-07-23-20-49-52.mp4')
clip = VideoFileClip(str(video_path))
# %%
clip.duration
# %%
frames: list[np.ndarray] = []
times = np.linspace(0, clip.duration, int(clip.duration) * 10)
# %%
for t in tqdm(times):
    frame = clip.get_frame(t)
    frames.append(frame)

# %%
left = 850
top = 400
width = 224

img = frames[60][top:top+width, left:left+width]
plt.imshow(img)
# %%
data = np.array([
    frame[top:top+width, left:left+width]
    for frame in tqdm(frames)
])
# %%
for i, frame in zip(times, tqdm(data)):
    cv2.imwrite(f'data/frames/{i:07.02f}.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
# %%
fit = umap.UMAP()
u = fit.fit_transform(data.reshape(-1, width*width*3))
# %%
plt.scatter(u[:,0], u[:,1])
# %%
tosave = {
    'imgs': data,
    'u': u,
}
open('chrs.pickle', 'wb').write(pickle.dumps(tosave))

# %%
del fit
# %%
u.shape