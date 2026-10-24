import hashlib,json
import numpy as np
import pytest
import trimesh
from gai.neural_edit import prepare_neural_edit,deformation_weights
from gai.local_editor import UnsupportedEdit
from gai.critic_schema import compile_choice,EditPlan
from gai.action_choice import decision_data


def test_diffusion_action_requires_compiled_region_and_cannot_claim_missing_geometry():
    with pytest.raises(ValueError):EditPlan(operation='neural_refine')
    proposal={'id':'region','permitted_operations':['neural_refine'],'regions':[{'reference_index':i,'bbox':[.1,.1,.8,.8]} for i in [0,1]]}
    choice=decision_data('M',{'M':.9},'region',['neural_refine'])
    result=compile_choice(choice,{'proposals':[proposal]},'a figurine',[])
    assert result['should_edit'] and result['edit_plan']['operation']=='neural_refine'
    assert result['edit_type']=='reshape'


def test_neural_job_rejects_stale_mesh_and_heldout_photo(tmp_path):
    mesh=tmp_path/'mesh.obj';mesh.write_text('mesh')
    evidence=tmp_path/'shape_evidence';evidence.mkdir()
    data={'mesh_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),'reserved_control':2,
          'views':[{'reference_index':i} for i in range(3)],'proposals':[{'id':'x','regions':[{'reference_index':0},{'reference_index':1}]}]}
    path=evidence/'catalog.json';path.write_text(json.dumps(data))
    _,_,training=prepare_neural_edit(mesh,tmp_path,'x')
    assert [v['reference_index'] for v in training]==[0,1]
    data['proposals'][0]['regions'][1]['reference_index']=2;path.write_text(json.dumps(data))
    with pytest.raises(UnsupportedEdit,match='control'):prepare_neural_edit(mesh,tmp_path,'x')
    mesh.write_text('changed')
    with pytest.raises(UnsupportedEdit,match='another mesh'):prepare_neural_edit(mesh,tmp_path,'x')


def test_deformation_is_zero_outside_patch_and_at_boundary():
    mesh=trimesh.creation.icosphere(subdivisions=3)
    indexes=np.flatnonzero(mesh.vertices[:,1]>.5)
    weights=deformation_weights(mesh,indexes)
    assert np.any(weights>0)
    exterior=np.ones(len(mesh.vertices),bool);exterior[indexes]=False
    assert np.all(weights[exterior]==0)
    boundary=[i for i in indexes if exterior[mesh.vertex_neighbors[i]].any()]
    assert np.all(weights[boundary]==0)
    with pytest.raises(UnsupportedEdit):deformation_weights(mesh,np.arange(len(mesh.vertices)))


def test_pipeline_passes_existing_mesh_to_neural_worker_without_reconstruction(tmp_path,monkeypatch):
    from gai import pipeline
    from gai.critic_schema import Critique
    source=tmp_path/'iteration_00';source.mkdir();mesh=source/'mesh.obj';trimesh.creation.icosphere(subdivisions=1).export(mesh)
    evidence=source/'shape_evidence';evidence.mkdir()
    proposal={'id':'patch','regions':[{'reference_index':i,'bbox':[.1,.1,.8,.8]} for i in [0,1]]}
    catalog={'mesh_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),'reserved_control':2,
             'views':[{'reference_index':i} for i in range(3)],'proposals':[proposal]}
    (evidence/'catalog.json').write_text(json.dumps(catalog))
    plan=EditPlan(operation='neural_refine',proposal_id='patch',evidence_basis='silhouette_catalog',regions=proposal['regions'],reason_en='Correct the shape.',reason_ru='Исправить форму.')
    critique=Critique(summary_ru='A shape defect.',camera_mismatch=False,confidence=.9,should_edit=True,edit_type='reshape',edit_plan=plan,issues=[{'part':'body'}],base_description='a figurine',target_description='a figurine')
    calls=[]
    def command(ident,args,**kwargs):
        calls.append(args);job=json.loads(args[-1].read_text());destination=tmp_path/'iteration_01/local_edit';destination.mkdir(parents=True)
        assert job['mesh']==str(mesh) and job['instruction']=='Correct the shape.'
        assert job['proposal_id']=='patch'
        import shutil
        shutil.copy2(mesh,destination/'mesh.obj')
    monkeypatch.setattr(pipeline,'command',command)
    cfg={'_edit_source':source,'neural_editor_enabled':True,'textdeformer_python':'python','neural_editor_model':'test','neural_editor_revision':'pinned'}
    path=pipeline.edit('fixture',mesh,tmp_path/'iteration_01',critique,cfg)
    assert path.is_file() and len(calls)==1
    assert calls[0][1].name=='neural_edit_worker.py'


def test_uninstalled_editor_is_rejected_before_a_run_is_created(monkeypatch,tmp_path):
    from gai.neural_edit import editor_config
    import gai.core
    monkeypatch.setattr(gai.core,'ROOT',tmp_path)
    cfg={'neural_editor_model':'test/model','neural_editor_revision':'pinned'}
    assert not editor_config(cfg,'standard')['neural_editor_enabled']
    with pytest.raises(ValueError,match='weights'):editor_config(cfg,'diffusion')
    with pytest.raises(ValueError,match='Unknown'):editor_config(cfg,'invented')
