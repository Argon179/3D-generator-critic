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


def test_multiview_keeps_holdout_out_of_reconstruction(tmp_path,monkeypatch):
    import gai.reconstruction as reconstruction
    root=tmp_path/'project';runs=root/'runs';runs.mkdir(parents=True)
    monkeypatch.setattr(core,'ROOT',root);monkeypatch.setattr(core,'RUNS',runs)
    monkeypatch.setattr(pipeline,'ROOT',root);monkeypatch.setattr(pipeline,'RUNS',runs)
    cfg=core.config();ident='abcdef654321';folder=runs/ident;folder.mkdir()
    refs=[{'path':str(folder/f'photo_{i}.png'),'azimuth_deg':a} for i,a in enumerate([0,90,180,270,330])]
    core.save_json(folder/'state.json',{'id':ident,'status':'queued','config':cfg,'parameters':{'seed':42,'background':'alpha','prompt':'a sphere','iterations':0,'generator':'multiview'},'references':refs,'iterations':[],'events':[]})
    def generate(run_id,args,**kwargs):
        job=json.loads(args[-1].read_text());out=__import__('pathlib').Path(job['out'])
        assert not job['retry']
        trimesh.creation.icosphere(subdivisions=1).export(out/'generated.glb')
        core.save_json(out/'references.json',[dict(r,prepared=r['path']) for r in refs])
    monkeypatch.setattr(pipeline,'command',generate)
    seen=[]
    def reconstruct(training):
        seen.extend(r['prepared'] for r in training)
        return trimesh.creation.icosphere(subdivisions=1)
    monkeypatch.setattr(reconstruction,'visual_hull',reconstruct)
    def render(run_id,mesh,current,cfg):
        core.save_json(current/'renders/views.json',[{'file':'view.png','azimuth_deg':0,'elevation_deg':0}])
    monkeypatch.setattr(pipeline,'render',render)
    monkeypatch.setattr(pipeline,'align',lambda *args:{'silhouette_iou':.9,'pairs':[{'reference_index':i,'iou':.9,'render':'view.png','boundary_f1':.9,'contour_distance':.01} for i in range(5)]})
    monkeypatch.setattr(pipeline,'criticize',lambda *args:pipeline.Critique(summary_ru='Synthetic fixture only',camera_mismatch=False,confidence=.9,should_edit=False,issues=[],base_description='a sphere',target_description='a sphere'))
    pipeline.execute(ident)
    state=core.read_run(ident)
    assert state['status']=='completed',state.get('error')
    assert len(seen)==4 and refs[1]['path'] not in seen
    assert state['iterations'][0]['metrics']['heldout']['count']==1
    assert (folder/'iteration_00/sf3d.obj').is_file()
    assert (folder/'iteration_00/reconstruction/mesh.obj').is_file()


def test_neural_multiview_bypasses_sf3d_and_marks_holdout(tmp_path,monkeypatch):
    from pathlib import Path
    root=tmp_path/'project';runs=root/'runs';runs.mkdir(parents=True)
    monkeypatch.setattr(core,'ROOT',root);monkeypatch.setattr(core,'RUNS',runs)
    monkeypatch.setattr(pipeline,'ROOT',root);monkeypatch.setattr(pipeline,'RUNS',runs)
    cfg=core.config();ident='abcdee123456';folder=runs/ident;folder.mkdir()
    refs=[{'path':str(folder/f'{i}.png'),'azimuth_deg':i*30} for i in range(12)]
    core.save_json(folder/'state.json',{'id':ident,'status':'queued','config':cfg,'parameters':{'seed':42,'background':'alpha','prompt':'a sphere','iterations':0,'generator':'hunyuan'},'references':refs,'iterations':[],'events':[]})
    def generate(run_id,args,**kwargs):
        assert str(args[-2]).endswith('multiview_worker.py')
        job=json.loads(args[-1].read_text());out=Path(job['out'])
        selected=set(job['selected'].values());assert selected=={0,3,6,9}
        trimesh.creation.icosphere(subdivisions=1).export(out/'generated.glb')
        core.save_json(out/'references.json',[dict(r,prepared=r['path'],used_for_generation=i in selected) for i,r in enumerate(refs)])
    monkeypatch.setattr(pipeline,'command',generate)
    monkeypatch.setattr(pipeline,'render',lambda *args:None)
    monkeypatch.setattr(pipeline,'align',lambda *args:{'silhouette_iou':.9,'pairs':[{'reference_index':i,'iou':.9,'render':'x.png'} for i in range(12)]})
    monkeypatch.setattr(pipeline,'criticize',lambda *args:pipeline.Critique(summary_ru='Fixture',camera_mismatch=False,confidence=.9,should_edit=False,base_description='a sphere',target_description='a sphere'))
    pipeline.execute(ident)
    state=core.read_run(ident)
    assert state['status']=='completed',state.get('error')
    assert state['iterations'][0]['backend']=='Hunyuan3D-2mv'
    assert state['iterations'][0]['metrics']['heldout']['count']==8
    assert not (folder/'iteration_00/sf3d.obj').exists()


def test_all_references_reach_critic_including_holdout(tmp_path,monkeypatch):
    from pathlib import Path
    folder=tmp_path/'iteration_00';folder.mkdir()
    core.save_json(folder/'renders/views.json',[])
    refs=[{'prepared':f'photo{i}.png'} for i in range(7)]
    metrics={'pairs':[{'reference_index':i,'render':f'{i}.png'} for i in range(7)]}
    def worker(run_id,args,**kwargs):
        payload=json.loads(Path(args[-1]).read_text())
        assert len(payload['pairs'])==7
        core.save_json(folder/'critique.json',{'summary_ru':'Fixture','camera_mismatch':False,'confidence':.8,
            'should_edit':False,'base_description':'a sphere','target_description':'a sphere','issues':[]})
    monkeypatch.setattr(pipeline,'command',worker)
    result=pipeline.criticize('fixture',folder,refs,metrics,'a sphere',core.config())
    assert result.edit_plan.operation=='none'


def test_local_candidate_needs_vlm_confirmation_even_if_iou_improves(tmp_path,monkeypatch):
    root=tmp_path/'project';runs=root/'runs';runs.mkdir(parents=True)
    monkeypatch.setattr(core,'ROOT',root);monkeypatch.setattr(core,'RUNS',runs)
    monkeypatch.setattr(pipeline,'ROOT',root);monkeypatch.setattr(pipeline,'RUNS',runs)
    cfg=core.config();ident='aaaabb112233';folder=runs/ident;folder.mkdir()
    core.save_json(folder/'state.json',{'id':ident,'status':'queued','config':cfg,
        'parameters':{'seed':42,'background':'alpha','prompt':'a sphere','iterations':1},
        'references':[{'path':'one.png'},{'path':'two.png'}],'iterations':[],'events':[]})
    def generate(run_id,args,**kwargs):
        from pathlib import Path
        job=json.loads(Path(args[-1]).read_text());out=Path(job['out'])
        trimesh.creation.icosphere(subdivisions=1).export(out/'generated.glb')
        core.save_json(out/'references.json',[{'prepared':'one.png'},{'prepared':'two.png'}])
    monkeypatch.setattr(pipeline,'command',generate)
    monkeypatch.setattr(pipeline,'render',lambda *args:None)
    def align(refs,render_dir,fixed=None):
        score=.7 if render_dir.parent.name=='iteration_00' else .9
        return {'silhouette_iou':score,'pairs':[{'reference_index':i,'iou':score,'render':'x.png'} for i in range(2)]}
    monkeypatch.setattr(pipeline,'align',align)
    critique=pipeline.Critique(summary_ru='Fixture',camera_mismatch=False,confidence=.9,should_edit=True,
        edit_type='reshape',edit_plan={'operation':'local_smooth'},view_assessments=[{'reference_index':0,'defects':[]}],issues=[{'part':'fixture'}],
        base_description='a sphere',target_description='a sphere')
    monkeypatch.setattr(pipeline,'criticize',lambda *args:critique)
    def edit(run_id,mesh,out,*args):
        candidate=trimesh.load(mesh);candidate.vertices[:,1]*=1.05
        path=out/'mesh.obj';candidate.export(path);return path
    monkeypatch.setattr(pipeline,'edit',edit)
    checked=[]
    def verify(*args):
        checked.append(True);return {'accepted':False,'reason_ru':'Synthetic new defect','views':[]}
    monkeypatch.setattr(pipeline,'verify_edit',verify)
    pipeline.execute(ident)
    state=core.read_run(ident)
    assert state['status']=='completed',state.get('error')
    assert checked and [i['accepted'] for i in state['iterations']]==[True,False]
    assert state['iterations'][1]['verification']['accepted'] is False
    assert np.isclose(trimesh.load(folder/'best.glb').extents.max(),2)


def test_failed_initial_critique_resumes_without_generation_or_render(tmp_path,monkeypatch):
    root=tmp_path/'project';runs=root/'runs';runs.mkdir(parents=True)
    monkeypatch.setattr(core,'ROOT',root);monkeypatch.setattr(core,'RUNS',runs)
    monkeypatch.setattr(pipeline,'ROOT',root);monkeypatch.setattr(pipeline,'RUNS',runs)
    cfg=core.config();ident='bbbbaa112233';folder=runs/ident;initial=folder/'iteration_00';initial.mkdir(parents=True)
    mesh=initial/'mesh.obj';trimesh.creation.icosphere(subdivisions=1).export(mesh)
    core.save_json(initial/'references.json',[{'prepared':'one.png'},{'prepared':'two.png'}])
    core.save_json(initial/'mesh_info.json',{'backend':'Hunyuan3D-2mv','can_deform':True})
    core.save_json(initial/'renders/views.json',[])
    core.save_json(folder/'state.json',{'id':ident,'status':'failed','stage':'critique','iteration':0,'config':cfg,
        'parameters':{'seed':42,'background':'alpha','prompt':'a sphere','iterations':0},
        'references':[],'preview_mesh':str(mesh.relative_to(root)),'iterations':[],'events':[],'error':'invalid JSON'})
    def forbidden(*args,**kwargs):raise AssertionError('Generation/render must not repeat')
    monkeypatch.setattr(pipeline,'command',forbidden);monkeypatch.setattr(pipeline,'render',forbidden)
    monkeypatch.setattr(pipeline,'align',lambda *args:{'silhouette_iou':.8,'pairs':[{'reference_index':i,'iou':.8,'render':'x.png'} for i in range(2)]})
    monkeypatch.setattr(pipeline,'criticize',lambda *args:pipeline.Critique(summary_ru='Test',camera_mismatch=False,confidence=.8,should_edit=False,base_description='a sphere',target_description='a sphere'))
    pipeline.execute(ident,resume_critique=True)
    state=core.read_run(ident)
    assert state['status']=='completed' and state['error'] is None,state.get('error')
    assert len(state['iterations'])==1 and (folder/'best.glb').exists()


def test_rejected_catalog_edit_can_try_a_different_vlm_operation(tmp_path,monkeypatch):
    from pathlib import Path
    root=tmp_path/'project';runs=root/'runs';runs.mkdir(parents=True)
    monkeypatch.setattr(core,'ROOT',root);monkeypatch.setattr(core,'RUNS',runs)
    monkeypatch.setattr(pipeline,'ROOT',root);monkeypatch.setattr(pipeline,'RUNS',runs)
    cfg=core.config();ident='ccccaa112233';folder=runs/ident;folder.mkdir()
    core.save_json(folder/'state.json',{'id':ident,'status':'queued','config':cfg,
        'parameters':{'seed':42,'background':'alpha','prompt':'an object','iterations':2},
        'references':[{'path':str(i)+'.png'} for i in range(3)],'iterations':[],'events':[]})
    def generate(run_id,args,**kwargs):
        job=json.loads(Path(args[-1]).read_text());out=Path(job['out'])
        trimesh.creation.icosphere(subdivisions=1).export(out/'generated.glb')
        core.save_json(out/'references.json',[{'prepared':str(i)+'.png'} for i in range(3)])
    monkeypatch.setattr(pipeline,'command',generate);monkeypatch.setattr(pipeline,'render',lambda *args:None)
    def align(refs,render_dir,fixed=None):
        score=.7 if render_dir.parent.name=='iteration_00' else .9
        return {'silhouette_iou':score,'pairs':[{'reference_index':i,'iou':score,'render':'x.png'} for i in range(3)]}
    monkeypatch.setattr(pipeline,'align',align)
    def critique(operation):
        return pipeline.Critique(summary_ru='Synthetic fixture',camera_mismatch=False,confidence=.8,should_edit=True,
            edit_type='reshape',edit_plan={'operation':operation,'proposal_id':'region_01','evidence_basis':'silhouette_catalog',
                'regions':[{'reference_index':i,'bbox':[.1,.1,.9,.9]} for i in [0,1]]},
            issues=[{'part':'fixture'}],view_assessments=[{'reference_index':0}],base_description='an object',target_description='an object')
    calls=[]
    def critic(run_id,source,refs,report,prompt,cfg):
        if '_excluded_edits' in cfg:
            assert ('fit_silhouette','region_01') in cfg['_excluded_edits'];calls.append('alternative')
            return critique('local_smooth')
        return critique('fit_silhouette')
    monkeypatch.setattr(pipeline,'criticize',critic)
    def edit(run_id,mesh,out,c,cfg):
        if c.edit_plan.operation=='local_smooth':
            assert Path(cfg['_edit_source']).name=='iteration_00'
        candidate=trimesh.load(mesh);candidate.vertices[:,1]*=1.02
        path=out/'mesh.obj';candidate.export(path);return path
    monkeypatch.setattr(pipeline,'edit',edit)
    monkeypatch.setattr(pipeline,'verify_edit',lambda run_id,before,after,*args:{'accepted':after.name=='iteration_02','reason_ru':'Synthetic fixture','views':[]})
    pipeline.execute(ident)
    state=core.read_run(ident)
    assert state['status']=='completed',state.get('error')
    assert [i['accepted'] for i in state['iterations']]==[True,False,True]
    assert state['iterations'][2]['edit_source_mesh']==state['iterations'][0]['mesh']
    assert calls==['alternative']
