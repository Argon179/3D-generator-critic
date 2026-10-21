"""CPU segmentation for approximately uniform dark/light backgrounds."""
import numpy as np
from PIL import Image
from scipy import ndimage


def uniform_background(image, threshold=28):
    a=np.array(image.convert('RGBA'))
    rgb=a[:,:,:3].astype(np.float32)
    # COIL backgrounds are dark grey (25 or 39), not RGB zero. Border outliers
    # include the object and a few acquisition artifacts; use a robust median.
    border=np.concatenate((rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]))
    background=np.median(border,axis=0)
    foreground=np.linalg.norm(rgb-background,axis=2)>threshold
    components,count=ndimage.label(foreground)
    if not count:raise ValueError('Не удалось выделить объект на однотонном фоне')
    sizes=np.bincount(components.ravel());sizes[0]=0
    mask=components==int(sizes.argmax())
    # For these simple figurines, retain interior dark eyes/paint as foreground.
    mask=ndimage.binary_fill_holes(mask)
    if mask.sum()<16 or mask.mean()>.97:
        raise ValueError('Ненадёжная маска: выберите U²-Net или подготовленный RGBA')
    a[:,:,3]=(mask*255).astype(np.uint8)
    return Image.fromarray(a)
