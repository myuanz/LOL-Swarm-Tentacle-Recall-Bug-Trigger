import numpy as np
from beartype import beartype
from dataclasses import dataclass

@beartype
@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int
    
def size_to_bbox[T: (int, float)](
    img_w: int, img_h: int,
    x: T, y: T, w: T, h: T
) -> BBox:
    types = [type(x), type(y), type(w), type(h)]
    assert all(t is int for t in types) or all(t is float for t in types), "All types should be either int or float"
    
    if isinstance(x, float):
        return BBox(
            int(x * img_w),
            int(y * img_h),
            int(w * img_w),
            int(h * img_w)
        )
    else:
        return BBox(x, y, w, h) # type: ignore

@beartype
def crop_image[T: (int, float)](
    image: np.ndarray, /, 
    x: T, y: T, w: T,
    *, 
    copy: bool = False
) -> np.ndarray:
    img_h, img_w = image.shape[:2]
    box = size_to_bbox(img_w, img_h, x, y, w, w)

    res = image[box.y:box.y + box.h, box.x:box.x + box.w]
    if copy:
        return res.copy()
    return res

