import numpy as np
from PIL import Image,ImageDraw
from gai.masks import uniform_background


def test_grey_background_and_sensor_outliers_do_not_become_foreground():
    image=Image.new('RGB',(128,128),(25,25,25));d=ImageDraw.Draw(image)
    d.ellipse((20,20,105,110),fill=(210,190,5));d.ellipse((50,40,55,45),fill=(25,25,25))
    d.point((0,0),fill=(53,51,32))
    a=np.array(uniform_background(image))[:,:,3]
    assert a[0,0]==0 and a[5,5]==0
    assert a[60,60]==255 and a[42,52]==255
    assert .3<(a>0).mean()<.6


def test_real_coil_duck_mask_has_no_opaque_border():
    from gai.core import ROOT
    image=Image.open(ROOT/'reference_images/multiview/figurines/rubber_duck/view_000deg.png')
    mask=np.array(uniform_background(image))[:,:,3]>0
    assert .2<mask.mean()<.8
    assert mask[0].mean()<.1 and mask[-1].mean()<.1
