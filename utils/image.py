import numpy as np
import matplotlib
from PIL import Image
matplotlib.use('agg')
import matplotlib.pyplot as plt


def mpl_to_np(fig):
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    buf = np.roll(buf, 3, axis=2)
    return buf


def mpl_to_pil(fig):
    buf = mpl_to_np(fig)
    w, h, d = buf.shape
    return Image.frombytes('RGBA', (w, h), buf.tostring())
