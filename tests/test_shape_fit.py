import hashlib,json
import numpy as np
import pytest
import trimesh
from PIL import Image
from gai.critic_schema import compile_choice
from gai.local_editor import raster_faces,UnsupportedEdit
from gai.shape_evidence import build_catalog
from gai.shape_fit import fit_mesh


def matrix(angle):
    a=np.radians(angle)
    return np.array([[np.cos(a),0,np.sin(a),0],[0,1,0,0],[-np.sin(a),0,np.cos(a),0],[0,0,0,1]])


def save_mask(path,mask,color=(100,100,100)):
    rgba=np.zeros((*mask.shape,4),np.uint8);rgba[:,:,:3]=color;rgba[:,:,3]=mask*255
    Image.fromarray(rgba).save(path)


def fixture(tmp_path):
    source=tmp_path/'iteration_00';(source/'renders').mkdir(parents=True)
    target=trimesh.creation.icosphere(subdivisions=2,radius=.75)
    mesh=target.copy();selected=mesh.vertices[:,0]>.3
    mesh.vertices[selected,0]+=.12
    path=source/'mesh.obj';mesh.export(path)
    records=[];views=[];pairs=[]
    for i,a in enumerate([0,55,120]):
        M=matrix(a)
        ref=source/f'target_{i}.png';save_mask(ref,raster_faces(target,M)>=0)
        render=source/'renders'/f'{i}.png';save_mask(render,raster_faces(mesh,M)>=0)
        records.append({'reference_index':i,'target':str(ref),'render':render.name,'projection_matrix':M.tolist()})
        views.append({'file':render.name,'projection_matrix':M.tolist(),'camera_location':[np.sin(np.radians(a)),0,np.cos(np.radians(a))],'azimuth_deg':a,'elevation_deg':0})
        pairs.append({'reference_index':i,'reference_path':str(ref),'render':render.name})
    (source/'renders/views.json').write_text(json.dumps(views))
    regions=[{'reference_index':i,'bbox':[.5,0,1,1]} for i in [0,1]]
    catalog={'mesh_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'views':records,'proposals':[{'id':'region_01','regions':regions,'vertex_indexes':np.flatnonzero(selected).tolist(),'operation':'fit_silhouette'}]}
    (source/'shape_evidence').mkdir();(source/'shape_evidence/catalog.json').write_text(json.dumps(catalog))
    return source,path,catalog,pairs


def test_shape_fit_improves_real_rasterized_shape_without_changing_topology(tmp_path):
    source,path,catalog,_=fixture(tmp_path)
    result=fit_mesh(path,source,tmp_path/'edit','region_01',steps=7)
    report=json.loads((result.parent/'edit_report.json').read_text())
    training=report['training_views']
    change=np.mean([report['after_fixed_frame_iou'][str(i)]-report['before_fixed_frame_iou'][str(i)] for i in training])
    assert change>.01
    a=trimesh.load(path,process=False);b=trimesh.load(result,process=False)
    assert np.array_equal(a.faces,b.faces)
    selected=np.array(catalog['proposals'][0]['vertex_indexes'])
    outside=np.setdiff1d(np.arange(len(a.vertices)),selected)
    assert np.allclose(a.vertices[outside],b.vertices[outside])
    assert report['control_views']==[2]


def test_color_changes_cannot_create_a_shape_proposal(tmp_path):
    source,path,_,pairs=fixture(tmp_path)
    for p in pairs:
        image=Image.open(source/'renders'/p['render']).convert('RGBA')
        a=np.asarray(image).copy();a[:,:,:3]=[250,30,20];Image.fromarray(a).save(p['reference_path'])
    assert build_catalog(path,source,pairs)['proposals']==[]


def test_vlm_can_only_choose_real_ids_and_can_request_no_change(tmp_path):
    _,_,catalog,_=fixture(tmp_path)
    data={'action':'fit_silhouette','proposal_id':'region_01','camera_mismatch':False,'confidence':.8,'reason_ru':'Supported contour mismatch'}
    critique=compile_choice(data,catalog,'an object',[])
    assert critique['edit_plan']['operation']=='fit_silhouette'
    assert critique['edit_plan']['regions']==catalog['proposals'][0]['regions']
    assert not compile_choice(dict(data,action='none'),catalog,'an object',[])['should_edit']
    with pytest.raises(ValueError):compile_choice(dict(data,proposal_id='invented'),catalog,'an object',[])


def test_stale_proposal_cannot_edit_a_different_mesh(tmp_path):
    source,path,_,_=fixture(tmp_path)
    path.write_text(path.read_text()+'\n# another mesh revision\n')
    with pytest.raises(UnsupportedEdit,match='другой версии'):fit_mesh(path,source,tmp_path/'edit','region_01')


def test_thin_engineering_plate_does_not_request_automatic_volume(tmp_path):
    source=tmp_path/'iteration_00';(source/'renders').mkdir(parents=True)
    plate=trimesh.creation.box(extents=[1.2,.8,.04]);path=source/'mesh.obj';plate.export(path)
    views=[];pairs=[]
    for i,a in enumerate([0,60,120]):
        M=matrix(a);mask=raster_faces(plate,M)>=0
        target=source/f'ref_{i}.png';render=source/'renders'/f'{i}.png'
        save_mask(target,mask,(230,180,40));save_mask(render,mask,(90,90,90))
        views.append({'file':render.name,'projection_matrix':M.tolist(),'camera_location':[np.sin(np.radians(a)),0,np.cos(np.radians(a))],'azimuth_deg':a,'elevation_deg':0})
        pairs.append({'reference_index':i,'reference_path':str(target),'render':render.name})
    (source/'renders/views.json').write_text(json.dumps(views))
    before=path.read_bytes()
    assert not build_catalog(path,source,pairs)['proposals']
    assert path.read_bytes()==before


def test_catalog_smoothing_preserves_outside_vertices_and_mesh_topology(tmp_path):
    from gai.shape_fit import smooth_mesh
    source,path,catalog,_=fixture(tmp_path)
    result=smooth_mesh(path,source,tmp_path/'smooth','region_01',iterations=5)
    a=trimesh.load(path,process=False);b=trimesh.load(result,process=False)
    outside=np.setdiff1d(np.arange(len(a.vertices)),catalog['proposals'][0]['vertex_indexes'])
    assert np.array_equal(a.faces,b.faces) and np.allclose(a.vertices[outside],b.vertices[outside])
    assert not np.allclose(a.vertices,b.vertices)
    assert b.is_watertight==a.is_watertight


def test_failed_operation_can_be_excluded_without_excluding_other_operations(tmp_path):
    _,_,catalog,_=fixture(tmp_path)
    catalog['proposals'][0]['permitted_operations']=['local_smooth']
    data={'action':'fit_silhouette','proposal_id':'region_01','camera_mismatch':False,'confidence':.8,'reason_ru':'Test geometric evidence'}
    with pytest.raises(ValueError):compile_choice(data,catalog,'an object',[])
    result=compile_choice(dict(data,action='local_smooth'),catalog,'an object',[])
    assert result['edit_plan']['operation']=='local_smooth'
