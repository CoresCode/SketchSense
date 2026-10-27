import numpy as np
from raster import rasterize, rasterize_uint8

# a crude "cat": head circle + ears + whiskers, drawn as polylines like QuickDraw
t = np.linspace(0, 2*np.pi, 40)
head = [ (128 + 60*np.cos(t)).tolist(), (150 + 55*np.sin(t)).tolist() ]
ear1 = [ [70,95,128],[80,40,55] ]
ear2 = [ [160,185,128],[60,40,55] ]
whisk = [ [40,110],[160,150] ]
strokes = [head, ear1, ear2, whisk]

img = rasterize(strokes, size=64)
print("shape", img.shape, "dtype", img.dtype, "min", img.min(), "max", img.max(), "mean", round(float(img.mean()),4))
u8 = rasterize_uint8(strokes, size=64)
print("uint8", u8.shape, u8.dtype, "nonzero px", int((u8>0).sum()))

# blank input must not crash
print("empty ->", rasterize([], size=64).sum())

# tiny dot
print("dot ->", rasterize([[[128],[128]]], size=64).sum() > 0)

# ascii render to eyeball it
chars = " .:-=+*#%@"
for row in img[::3]:
    print("".join(chars[min(int(v*9),9)] for v in row))
