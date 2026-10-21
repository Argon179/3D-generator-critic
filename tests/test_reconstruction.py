import numpy as np
import pytest
from PIL import Image,ImageDraw
from gai.reconstruction import visual_hull
from gai.geometry import geometry_report


def test_known_views_recover_volume_not_a_billboard(tmp_path):
    path=tmp_path/'circle.png';im=Image.new('RGBA',(96,96));ImageDraw.Draw(im).ellipse((12,12,83,83),fill='white');im.save(path)
    refs=[{'prepared':str(path),'azimuth_deg':a} for a in (0,90,180,270)]
    mesh=visual_hull(refs,resolution=48)
    report=geometry_report(mesh)
    assert report['thickness_ratio']>.7
    assert mesh.is_watertight
    assert np.min(mesh.extents)/np.max(mesh.extents)>.9
    with pytest.raises(ValueError,match='азимута'):visual_hull([dict(r,azimuth_deg=None) for r in refs])
    with pytest.raises(ValueError,match='покрытие'):visual_hull([dict(r,azimuth_deg=i*10) for i,r in enumerate(refs)])
