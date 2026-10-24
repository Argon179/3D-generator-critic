from fastapi.testclient import TestClient
from gai.server import app


def test_library_and_files_are_real_and_confined():
    client=TestClient(app)
    datasets=client.get('/api/library').json()
    assert len(datasets)>=6
    assert all(len(x['images'])>=2 for x in datasets)
    first=datasets[0]['images'][0]['path']
    response=client.get('/api/files/'+first)
    assert response.status_code==200
    assert response.headers['content-type'].startswith('image/')
    for path in ['.cache/huggingface/token','configs/project.json','runs/../../configs/project.json']:
        assert client.get('/api/files/'+path).status_code!=200


def test_input_rejects_unknown_dataset_and_cross_origin():
    client=TestClient(app)
    payload={'dataset_id':'../../etc','image_indexes':[0,1],'prompt':'a duck'}
    assert client.post('/api/runs',json=payload).status_code==404
    assert client.post('/api/runs',json=payload,headers={'Origin':'https://example.com'}).status_code==403


def test_generation_view_and_roi_validated():
    client=TestClient(app);dataset=client.get('/api/library').json()[0]
    body={'dataset_id':dataset['id'],'image_indexes':[0,1],'generation_index':2,'prompt':'a duck'}
    assert client.post('/api/runs',json=body).status_code==400
    body.update(generation_index=0,crop=[.8,.1,.5,.5])
    assert client.post('/api/runs',json=body).status_code==400


def test_stage_preview_cannot_read_outside_the_run():
    client=TestClient(app)
    response=client.get('/api/runs/abcdef123456/preview',params={'artifact':'configs/project.json'})
    assert response.status_code==404
    assert client.get('/api/runs/not-a-run/stages').status_code==404


def test_existing_run_stage_inventory_and_glb_preview():
    client=TestClient(app)
    # Saved local regression run; fresh clones can still run all synthetic checks.
    from gai.core import RUNS
    import pytest
    ident='1445bcda34e7'
    if not (RUNS/ident/'state.json').exists():pytest.skip('No saved local run')
    result=client.get(f'/api/runs/{ident}/stages')
    assert result.status_code==200
    stages=result.json()['stages']
    assert {'inputs','generation','critic_0','render_0'}<={s['id'] for s in stages}
    generated=f'runs/{ident}/iteration_00/generated.glb'
    preview=client.get(f'/api/runs/{ident}/preview',params={'artifact':generated})
    assert preview.status_code==200 and '\nv ' in preview.text and '\nf ' in preview.text
    assert client.get(f'/api/runs/{ident}/preview',params={'artifact':'runs/../configs/project.json'}).status_code==404


def test_manual_sides_accept_three_photos_without_known_angles(tmp_path,monkeypatch):
    import gai.server as server
    import gai.core as core
    from PIL import Image
    from types import SimpleNamespace
    monkeypatch.setattr(core,'RUNS',tmp_path/'runs')
    monkeypatch.setattr(server,'ROOT',tmp_path)
    images=[]
    for index in range(3):
        path=tmp_path/f'photo_{index}.png';Image.new('RGBA',(8,8)).save(path)
        images.append({'path':path.name,'azimuth_deg':None})
    monkeypatch.setattr(server,'library',lambda:[{'id':'manual','name':'Manual photos','images':images}])
    monkeypatch.setattr(server,'weight_checks',lambda:[])
    monkeypatch.setattr(core,'multiview_checks',lambda backend:[])
    submitted=[]
    monkeypatch.setattr(server,'executor',SimpleNamespace(submit=lambda *args:submitted.append(args)))
    result=TestClient(app).post('/api/runs',json={'dataset_id':'manual','image_indexes':[0,1,2],
        'generation_index':0,'prompt':'a rubber duck','generator':'hunyuan','background':'alpha',
        'view_sides':{'front':0,'left':1}})
    assert result.status_code==200,result.text
    state=result.json()
    assert state['parameters']['view_sides']=={'front':0,'left':1}
    assert all(r['azimuth_deg'] is None for r in state['references'])
    assert len(submitted)==1 and state['status']=='queued'


def test_refine_endpoint_queues_existing_mesh_without_generation(monkeypatch):
    import gai.server as server
    import gai.refinement as refinement
    from types import SimpleNamespace
    monkeypatch.setattr(server,'weight_checks',lambda:[])
    monkeypatch.setattr(refinement,'create_refinement',lambda ident,iterations:{'id':'deedee112233','parent_run':ident,'status':'queued','iterations_requested':iterations})
    submitted=[]
    monkeypatch.setattr(server,'executor',SimpleNamespace(submit=lambda *args,**kwargs:submitted.append((args,kwargs))))
    result=TestClient(app).post('/api/runs/36312ae42e91/refine',json={'iterations':2})
    assert result.status_code==200,result.text
    assert result.json()['parent_run']=='36312ae42e91'
    assert submitted[0][1]['resume_critique'] is True
    assert TestClient(app).post('/api/runs/36312ae42e91/refine',json={'iterations':0}).status_code==422
