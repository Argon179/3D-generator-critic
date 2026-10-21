"""Orchestration test with explicit fake GPU boundaries, never a product demo."""
import json
import numpy as np
import trimesh
import gai.core as core
import gai.pipeline as pipeline


def test_rejected_edit_preserves_previous_best_mesh(tmp_path,monkeypatch):
    cfg=core.config()
    root=tmp_path/'project';runs=root/'runs';runs.mkdir(parents=True)
    monkeypatch.setattr(core,'ROOT',root);monkeypatch.setattr(core,'RUNS',runs)
    monkeypatch.setattr(pipeline,'ROOT',root);monkeypatch.setattr(pipeline,'RUNS',runs)
    monkeypatch.setattr(pipeline,'config',lambda:cfg)
    ident='abcdef123456';folder=runs/ident;folder.mkdir()
    core.save_json(folder/'state.json',{'id':ident,'status':'queued','parameters':{'seed':42,'background':'alpha','prompt':'a sphere','iterations':1},'references':[{'path':'one.png'},{'path':'two.png'}],'iterations':[],'events':[]})
    def fake_generate(run_id,args,**kwargs):
        job=json.loads(args[-1].read_text());out=__import__('pathlib').Path(job['out'])
        trimesh.creation.icosphere(subdivisions=1).export(out/'generated.glb')
        core.save_json(out/'references.json',[{'prepared':'one.png'},{'prepared':'two.png'}])
    monkeypatch.setattr(pipeline,'command',fake_generate)
    monkeypatch.setattr(pipeline,'render',lambda *args:None)
    def fake_align(refs,render_dir,fixed=None):
        score=.8 if render_dir.parent.name=='iteration_00' else .5
        return {'silhouette_iou':score,'pairs':[{'reference_index':i,'iou':score,'render':'x.png'} for i in range(2)]}
    monkeypatch.setattr(pipeline,'align',fake_align)
    critique=pipeline.Critique(summary_ru='Test fixture only',camera_mismatch=False,confidence=.9,should_edit=True,issues=[{'part':'fixture','observation_ru':'Test fixture mismatch','suggestion_ru':'Test fixture change'}],base_description='a sphere',target_description='an oval sphere')
    monkeypatch.setattr(pipeline,'criticize',lambda *args:critique)
    def fake_edit(run_id,mesh,next_folder,*args):
        candidate=trimesh.load(mesh,force='mesh');candidate.vertices[:,1]*=1.2
        p=next_folder/'edited.obj';candidate.export(p);return p
    monkeypatch.setattr(pipeline,'edit',fake_edit)
    pipeline.execute(ident)
    state=core.read_run(ident)
    assert state['status']=='completed',state.get('error')
    assert [i['accepted'] for i in state['iterations']]==[True,False]
    best=trimesh.load(folder/'best.glb',force='mesh')
    assert np.isclose(best.extents.max(),2)
    assert 'отклонена' in state['events'][-2]['message']
