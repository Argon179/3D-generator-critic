import numpy as np
import pytest
from PIL import Image,ImageDraw
from gai.core import save_json
from gai.metrics import normalized_mask,iou,align,acceptance


def rgba(path,box):
    im=Image.new('RGBA',(100,100));ImageDraw.Draw(im).rectangle(box,fill='white');im.save(path)


def test_background_requires_actual_mask(tmp_path):
    p=tmp_path/'opaque.png';Image.new('RGB',(10,10),'white').save(p)
    with pytest.raises(ValueError,match='маски'):normalized_mask(p)


def test_translation_scale_invariance_but_shape_sensitivity(tmp_path):
    a=tmp_path/'a.png';b=tmp_path/'b.png';c=tmp_path/'c.png'
    rgba(a,(10,10,39,69));rgba(b,(50,20,64,49));rgba(c,(10,10,69,39))
    assert iou(normalized_mask(a),normalized_mask(b))>.95
    assert iou(normalized_mask(a),normalized_mask(c))<.4


def test_shared_camera_and_fixed_correspondences(tmp_path):
    for name,box in [('front.png',(20,10,39,79)),('side.png',(10,20,79,39))]:rgba(tmp_path/name,box)
    save_json(tmp_path/'views.json',[{'file':'front.png','azimuth_deg':0,'elevation_deg':20},{'file':'side.png','azimuth_deg':180,'elevation_deg':20}])
    refs=[{'prepared':str(tmp_path/'side.png'),'azimuth_deg':0},{'prepared':str(tmp_path/'front.png'),'azimuth_deg':180}]
    # Reference files are immutable while candidate render files change.
    for i,r in enumerate(refs):
        p=tmp_path/f'ref_{i}.png';p.write_bytes(__import__('pathlib').Path(r['prepared']).read_bytes());r['prepared']=str(p)
    first=align(refs,tmp_path)
    assert first['silhouette_iou']==1
    assert first['pairs'][0]['azimuth_deg']==180
    # If edited silhouettes swap, fixed camera pairing must penalize it instead of rematching.
    rgba(tmp_path/'side.png',(20,10,39,79));rgba(tmp_path/'front.png',(10,20,79,39))
    after=align(refs,tmp_path,first['pairs'])
    assert after['silhouette_iou']<.4


def test_edit_cannot_win_by_only_improving_generation_image():
    report=lambda values:{'pairs':[{'reference_index':i,'iou':v} for i,v in enumerate(values)]}
    assert not acceptance(report([.5,.8,.8]),report([1,.7,.7]))[0]
    assert acceptance(report([.5,.6,.6]),report([.5,.7,.7]))[0]
    assert not acceptance(report([.5,.7,.6]),report([.5,.66,.8]))[0]


def test_top_audit_cannot_replace_reference_camera(tmp_path):
    rgba(tmp_path/'reference.png',(10,20,79,39))
    rgba(tmp_path/'front.png',(20,10,39,79))
    rgba(tmp_path/'top.png',(10,20,79,39))
    save_json(tmp_path/'views.json',[{'file':'front.png','azimuth_deg':0,'elevation_deg':0},{'file':'top.png','azimuth_deg':0,'elevation_deg':90}])
    report=align([{'prepared':str(tmp_path/'reference.png')}],tmp_path)
    assert report['pairs'][0]['render']=='front.png'
    assert report['silhouette_iou']<.4


def test_retry_photos_are_not_holdout_evidence():
    def report(values):
        return {'pairs':[{'reference_index':i,'iou':v,'used_for_generation':i in (0,1,2)} for i,v in enumerate(values)]}
    # Improved on every attempted photo, worse on the only unseen photo.
    assert not acceptance(report([.5,.5,.5,.8]),report([.9,.9,.9,.7]))[0]
    assert not acceptance(report([.5,.5,.5]),report([.9,.9,.9]))[0]
