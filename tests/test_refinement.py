import json
import numpy as np
import pytest
import trimesh
from PIL import Image
from gai.core import save_json


def test_refinement_clones_accepted_coordinates_and_uses_current_critic_config(tmp_path,monkeypatch):
    import gai.refinement as refinement
    root=tmp_path;runs=root/'runs';ident='abcabc112233';folder=runs/ident;initial=folder/'iteration_00';initial.mkdir(parents=True)
    mesh=trimesh.creation.box(extents=[1,2,1]);mesh.apply_translation([.1,.2,0]);mesh.export(initial/'mesh.obj')
    refs=[]
    for i in range(3):
        raw=folder/'source_images'/f'{i}.png';raw.parent.mkdir(exist_ok=True);Image.new('RGBA',(10,10),'white').save(raw)
        dest=initial/f'reference_{i:02d}.png';Image.new('RGBA',(12,12),'white').save(dest)
        refs.append({'path':str(raw),'prepared':str(dest),'azimuth_deg':i*90,'used_for_generation':i<2})
    save_json(initial/'references.json',refs);save_json(initial/'renders/views.json',[])
    parent={'id':ident,'name':'Test engineering part','status':'completed','parameters':{'iterations':1,'prompt':'a part'},
        'references':[{'path':r['path'],'azimuth_deg':r['azimuth_deg']} for r in refs],
        'iterations':[{'number':0,'accepted':True,'mesh':str((initial/'mesh.obj').relative_to(root)),'path':str(initial.relative_to(root)),'backend':'fixture'}]}
    monkeypatch.setattr(refinement,'ROOT',root);monkeypatch.setattr(refinement,'RUNS',runs)
    monkeypatch.setattr(refinement,'read_run',lambda _:parent);monkeypatch.setattr(refinement,'run_dir',lambda _:folder)
    monkeypatch.setattr(refinement,'config',lambda:{'qwen_model':'current-test-model'})
    state=refinement.create_refinement(ident,iterations=2);out=runs/state['id']
    assert state['status']=='queued' and state['parent_run']==ident and state['initial_mesh_import']
    assert state['config']['qwen_model']=='current-test-model' and state['parameters']['iterations']==2
    a=trimesh.load(initial/'mesh.obj',process=False);b=trimesh.load(out/'iteration_00/mesh.obj',process=False)
    assert np.array_equal(a.vertices,b.vertices) and np.array_equal(a.faces,b.faces)
    assert all(str(out) in r['path'] for r in state['references'])
    assert (out/'iteration_00/renders/views.json').exists()
    assert json.loads((out/'iteration_00/references.json').read_text())[0]['used_for_generation']
