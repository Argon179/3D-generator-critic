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
