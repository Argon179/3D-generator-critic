import json
import numpy as np
import pytest
import trimesh
from PIL import Image
from gai.core import save_json
from gai.critic_schema import EditPlan
from gai.local_editor import apply_plan,raster_faces,UnsupportedEdit
from gai.review import summarize_review


def fixture(tmp_path):
    mesh=trimesh.creation.icosphere(subdivisions=3,radius=.9)
    source=tmp_path/'source';(source/'renders').mkdir(parents=True)
    path=source/'mesh.obj';mesh.export(path)
    front=np.eye(4);side=np.array([[0,0,-1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
    views=[]
    for i,matrix in enumerate([front,side]):
        Image.new('RGBA',(192,192),'white').save(source/'renders'/f'{i}.png')
        views.append({'file':f'{i}.png','projection_matrix':matrix.tolist()})
    save_json(source/'renders/views.json',views)
    save_json(source/'metrics.json',{'pairs':[{'reference_index':i,'render':f'{i}.png'} for i in range(2)]})
    regions=[{'reference_index':i,'bbox':[.7,.25,.95,.75]} for i in range(2)]
    return source,path,regions


def test_zbuffer_does_not_select_occluded_faces():
    mesh=trimesh.Trimesh(vertices=[[-.8,-.8,-.5],[.8,-.8,-.5],[0,.8,-.5],[-.8,-.8,.5],[.8,-.8,.5],[0,.8,.5]],faces=[[0,1,2],[3,4,5]],process=False)
    ids=raster_faces(mesh,np.eye(4),size=32)
    assert 0 in ids and 1 not in ids


def test_local_deformation_is_bounded_and_keeps_unselected_vertices(tmp_path):
    source,path,regions=fixture(tmp_path)
    out=tmp_path/'edit'
    result=apply_plan(path,EditPlan(operation='local_scale',scale=1.08,regions=regions),source,out)
    a=trimesh.load(path,process=False);b=trimesh.load(result,process=False)
    changed=np.linalg.norm(a.vertices-b.vertices,axis=1)
    assert changed.max()>0 and changed.max()<.0541
    assert np.count_nonzero(changed)==np.count_nonzero(changed>1e-8)
    assert np.count_nonzero(changed)<len(a.vertices)//3
    assert np.array_equal(a.faces,b.faces)


def test_removal_changes_topology_instead_of_inflating(tmp_path):
    source,path,regions=fixture(tmp_path)
    out=tmp_path/'edit'
    result=apply_plan(path,EditPlan(operation='remove_region',regions=regions),source,out)
    a=trimesh.load(path);b=trimesh.load(result)
    assert 0<len(a.faces)-len(b.faces)<len(a.faces)*.2
    assert set(map(tuple,np.round(b.vertices,7))) <= set(map(tuple,np.round(a.vertices,7)))
    assert (out/'region_00.png').exists()


def test_one_view_or_repeated_camera_cannot_authorize_removal(tmp_path):
    source,path,regions=fixture(tmp_path)
    with pytest.raises(UnsupportedEdit,match='двух'):
        apply_plan(path,EditPlan(operation='remove_region',regions=regions[:1]),source,tmp_path/'a')
    save_json(source/'metrics.json',{'pairs':[{'reference_index':i,'render':'0.png'} for i in range(2)]})
    with pytest.raises(UnsupportedEdit,match='разные камеры'):
        apply_plan(path,EditPlan(operation='remove_region',regions=regions),source,tmp_path/'b')


def test_repair_does_not_fill_legitimate_holes(tmp_path):
    mesh=trimesh.creation.box();mesh.update_faces(np.arange(len(mesh.faces))!=0)
    mesh.faces=np.vstack([mesh.faces,mesh.faces[0]])
    path=tmp_path/'source.obj';mesh.export(path)
    result=apply_plan(path,EditPlan(operation='repair_mesh'),tmp_path,tmp_path/'edit')
    fixed=trimesh.load(result)
    assert not fixed.is_watertight and len(fixed.faces)==11


def review(i,**kw):
    return dict(reference_index=i,camera_mismatch=False,improved=True,worsened=False,confidence=.8,explanation_ru='Synthetic test',**kw)


def test_review_needs_two_improved_views_and_no_new_defect():
    a=review(0);b=review(1)
    assert summarize_review([a,b],[0,1])['accepted']
    assert not summarize_review([a],[0,1])['accepted']
    assert not summarize_review([a,dict(b,improved=False)],[0,1])['accepted']
    assert not summarize_review([a,b,dict(review(2),worsened=True,confidence=.1)],[0,1])['accepted']
    with pytest.raises(ValueError):summarize_review([dict(a,improved='true'),b],[0,1])


def test_verifier_is_not_primed_with_critic_verdict():
    from gai.review import verification_context
    context=verification_context({'summary_ru':'This is definitely improved','confidence':.99,
        'edit_plan':{'operation':'fit_silhouette','reason_ru':'You must agree with the critic','regions':[{'reference_index':2}]}})
    text=json.dumps(context)
    assert 'definitely' not in text and 'must agree' not in text and 'confidence' not in text
    assert context['operation']=='fit_silhouette' and context['reference_indexes']==[2]


def test_blind_review_unblinds_counterbalanced_order_and_rejects_new_defects():
    from gai.review import unblind_review
    data={'preferred':'A','camera_mismatch':False,'a_has_unique_defect':False,'b_has_unique_defect':True,'confidence':.9,'explanation_ru':'Visible difference'}
    assert unblind_review(data,1,'A')['improved']
    assert not unblind_review(data,1,'A')['worsened']
    assert unblind_review(data,0,'B')['worsened']
    assert not unblind_review(data,0,'B')['improved']
    data.update(preferred='tie',b_has_unique_defect=False)
    assert not unblind_review(data,0,'B')['improved']
    assert not unblind_review(data,0,'B')['worsened']
