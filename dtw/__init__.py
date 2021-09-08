import numpy as np
from .dtw.core import dtw_c, dtw_c_interp


def dtw(x, y=None, interp=False):
    # x : src mfcc, [Time, Dim]
    # y : tgt mfcc, [Time, Dim]
    # interp : whether interpolate frame or not

    if y is None:
        dist = x
    else:
        dist = np.sqrt(
            np.sum(x ** 2, axis=1, keepdims=True)
            - 2 * x @ y.transpose(1, 0)
            + np.sum(y.transpose(1, 0) ** 2, axis=0, keepdims=True)
        )
    path = np.zeros_like(dist).astype(np.float32)
    mask1 = np.zeros_like(dist).astype(np.int32)
    mask2 = np.zeros_like(dist).astype(np.int32)
    t_x = path.shape[0]
    t_y = path.shape[1]
    if interp:
        dtw_c_interp(path, dist, mask1, mask2, t_x, t_y, 1e+9)
    else:
        dtw_c(path, dist, mask1, mask2, t_x, t_y, 1e+9)
    return path
