import numpy as np
from beartype import beartype


@beartype
def crop_image[T: (int, float)](
    image: np.ndarray, /, 
    x: T, y: T, w: T,
    *, 
    copy: bool = False
) -> np.ndarray:
    img_h, img_w = image.shape[:2]
    if isinstance(x, float):
        _x = int(x * img_w)
        _y = int(y * img_h)
        _w = int(w * img_w)
    else:
        _x = x
        _y = y
        _w = w

    res = image[_y:_y+_w, _x:_x+_w]
    if copy:
        return res.copy()
    return res

