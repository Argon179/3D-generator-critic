from __future__ import annotations

import io
import json
import uuid
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from gai.core import ROOT, RUNS, config, doctor, event, library, now, read_run, run_dir, save_json, update_run, weight_checks
from gai.pipeline import execute

executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='gai-gpu')
diagnostics=ThreadPoolExecutor(max_workers=1,thread_name_prefix='gai-doctor')
doctor_job=None


@asynccontextmanager
async def lifespan(app):
    RUNS.mkdir(exist_ok=True)
    for path in RUNS.glob('*/state.json'):
        state=json.loads(path.read_text())
        if state['status'] in {'running','queued'}:
            state.update(status='interrupted',error='Приложение было закрыто. Создайте новый запуск.');save_json(path,state)
    yield
    for path in RUNS.glob('*/state.json'):
        try:
            state=json.loads(path.read_text())
            if state['status'] in {'queued','running'}:(path.parent/'cancel').touch()
        except (OSError,ValueError):pass


app=FastAPI(title='GAI · 3D Loop',lifespan=lifespan)


@app.middleware('http')
async def same_origin(request:Request,call_next):
    # Local application: disallow DNS rebinding and cross-origin mutation.
    if request.url.hostname not in {'127.0.0.1','localhost','::1','testserver'}:
        return JSONResponse({'detail':'Только локальный доступ'},status_code=403)
    if request.method in {'POST','PUT','DELETE','PATCH'}:
        origin=request.headers.get('origin')
        if origin and origin!=str(request.base_url).rstrip('/'):
            return JSONResponse({'detail':'Запрос из другого origin запрещён'},status_code=403)
    return await call_next(request)


@app.get('/api/library')
def get_library():return library()


@app.get('/api/doctor')
def get_doctor(refresh:bool=False):
    global doctor_job
    if doctor_job is None or (refresh and doctor_job.done()):
        doctor_job=diagnostics.submit(doctor)
    return doctor_job.result() if doctor_job.done() else {'pending':True}


class RunRequest(BaseModel):
    dataset_id:str
    image_indexes:list[int]=Field(min_length=2,max_length=12)
    generation_index:int=Field(default=0,ge=0)
    prompt:str=Field(min_length=3,max_length=200)
    seed:int=Field(default=42,ge=0,le=2**31-1)
    iterations:int=Field(default=1,ge=0,le=3)
    background:str='rembg'
    generator:str='auto'
    editor:Literal['standard','diffusion']='standard'
    view_sides:dict[str,int]|None=None
    crop:list[float]|None=None


label_jobs={}

@app.post('/api/auto-label')
def auto_label(body:RunRequest):
    dataset=next((d for d in library() if d['id']==body.dataset_id),None)
    if dataset is None:raise HTTPException(404,'Набор не найден')
    indexes=list(dict.fromkeys(body.image_indexes))
    if len(indexes)<3 or any(i<0 or i>=len(dataset['images']) for i in indexes):
        raise HTTPException(400,'Выберите минимум 3 фото одного объекта')
    if len(label_jobs)>=100:
        for key in list(label_jobs):
            if label_jobs[key].done():del label_jobs[key]
    if len(label_jobs)>=100:raise HTTPException(429,'Очередь авторазметки заполнена')
    from gai.autolabel import infer_labels
    ident=uuid.uuid4().hex
    refs=[{'index':i,'path':str(ROOT/dataset['images'][i]['path'])} for i in indexes]
    label_jobs[ident]=executor.submit(infer_labels,ident,refs)
    return {'id':ident}

@app.get('/api/auto-label/{ident}')
def label_status(ident:str):
    future=label_jobs.get(ident)
    if future is None:raise HTTPException(404,'Задание не найдено')
    if not future.done():return {'status':'running'}
    try:return {'status':'completed',**future.result()}
    except Exception as exc:return {'status':'failed','error':str(exc)}


@app.post('/api/runs')
def create_run(body:RunRequest):
    dataset=next((x for x in library() if x['id']==body.dataset_id),None)
    if dataset is None:raise HTTPException(404,'Набор не найден')
    indexes=list(dict.fromkeys(body.image_indexes))
    if len(indexes)<2 or any(i<0 or i>=len(dataset['images']) for i in indexes):raise HTTPException(400,'Выберите минимум два разных ракурса')
    if body.generation_index not in indexes:raise HTTPException(400,'Фото для генерации должно быть выбрано')
    if body.generator not in {'auto','sf3d','multiview','hunyuan','da3'}:raise HTTPException(400,'Неизвестный генератор')
    if body.generator=='multiview' and (len(indexes)<5 or any(dataset['images'][i].get('azimuth_deg') is None for i in indexes)):
        raise HTTPException(400,'Для многовидового режима нужны минимум 5 фото с известными азимутами')
    if body.background not in {'black','white','rembg','alpha'}:raise HTTPException(400,'Неизвестный режим фона')
    if body.crop is not None:
        if len(body.crop)!=4:raise HTTPException(400,'ROI: x,y,width,height')
        x,y,w,h=body.crop
        if not all(0<=v<=1 for v in body.crop) or w<=0 or h<=0 or x+w>1 or y+h>1:raise HTTPException(400,'ROI должен лежать внутри изображения')
    ordered=[body.generation_index]+[i for i in indexes if i!=body.generation_index]
    if body.generator in {'hunyuan','da3'}:
        from gai.multiview import select_views
        try:select_views([dataset['images'][i] for i in ordered],body.generator,body.view_sides)
        except ValueError as exc:raise HTTPException(400,str(exc))
        from gai.core import multiview_checks
        missing=[x['name'] for x in multiview_checks(body.generator) if not x['ok']]
        if missing:raise HTTPException(409,'Генератор ещё не готов: '+', '.join(missing)+'. См. scripts/setup_multiview.py')
    missing=[x['name'] for x in weight_checks() if not x['ok'] and (body.generator not in {'hunyuan','da3'} or x['name']!='Веса SF3D')]
    if missing:raise HTTPException(409,'Веса ещё не полностью загружены: '+', '.join(missing)+'. Дождитесь завершения загрузки; инструкция — в README.md.')
    ordered=[body.generation_index]+[i for i in indexes if i!=body.generation_index]
    refs=[dict(dataset['images'][i],path=str(ROOT/dataset['images'][i]['path']),crop=body.crop) for i in ordered]
    from gai.neural_edit import editor_config
    try:run_config=editor_config(config(),body.editor)
    except ValueError as exc:raise HTTPException(400,str(exc)) from exc
    ident=uuid.uuid4().hex[:12];folder=run_dir(ident);folder.mkdir(parents=True)
    inputs=folder/'source_images';inputs.mkdir()
    for i,ref in enumerate(refs):
        source=Path(ref['path']);dest=inputs/f'view_{i:02d}{source.suffix.lower()}'
        shutil.copy2(source,dest);ref.update(original_path=ref['path'],path=str(dest))
    state={'id':ident,'name':dataset['name'],'created_at':now(),'updated_at':now(),'status':'queued','stage':'queued','parameters':body.model_dump(),'config':run_config,'references':refs,'iterations':[],'events':[]}
    save_json(folder/'state.json',state);event(ident,'Запуск поставлен в очередь')
    executor.submit(execute,ident)
    return read_run(ident)


@app.get('/api/runs')
def runs():
    result=[]
    for p in RUNS.glob('*/state.json'):
        try:
            state=read_run(p.parent.name)
            result.append({k:state.get(k) for k in ['id','name','status','stage','created_at','updated_at']})
        except (OSError,ValueError):continue
    return sorted(result,key=lambda x:x['created_at'],reverse=True)


@app.get('/api/runs/{ident}')
def get_run(ident:str):
    try:return read_run(ident)
    except (OSError,ValueError):raise HTTPException(404,'Запуск не найден')


class RefineRequest(BaseModel):
    iterations:int=Field(default=3,ge=1,le=3)
    editor:Literal['standard','diffusion']='standard'


@app.post('/api/runs/{ident}/refine')
def refine_saved_run(ident:str,body:RefineRequest):
    from gai.refinement import create_refinement
    try:
        problems=[c for c in weight_checks() if not c['ok'] and 'qwen' in c.get('name','').lower()]
        if problems:raise ValueError('Веса критика ещё не установлены')
        state=create_refinement(ident,body.iterations,editor=body.editor)
    except (ValueError,FileNotFoundError) as exc:raise HTTPException(400,str(exc)) from exc
    executor.submit(execute,state['id'],resume_critique=True)
    return state


@app.get('/api/runs/{ident}/stages')
def get_stages(ident:str):
    from gai.artifacts import stage_inventory
    try:return stage_inventory(ident)
    except (OSError,ValueError):raise HTTPException(404,'Запуск не найден')


@app.get('/api/runs/{ident}/preview')
def get_mesh_preview(ident:str,artifact:str):
    import numpy as np
    import trimesh
    try:folder=run_dir(ident).resolve()
    except ValueError:raise HTTPException(404,'Запуск не найден')
    path=(ROOT/artifact).resolve()
    if not path.is_relative_to(folder) or not path.is_file() or path.suffix not in {'.obj','.glb'}:
        raise HTTPException(404,'Модель не найдена')
    mesh=trimesh.load(path,force='mesh',process=True)
    if len(mesh.faces)>20000:mesh=mesh.simplify_quadric_decimation(face_count=20000)
    span=float(np.ptp(mesh.vertices,axis=0).max())
    if span<=0:raise HTTPException(422,'Пустая геометрия')
    mesh.vertices=(mesh.vertices-mesh.bounds.mean(axis=0))*2/span
    return PlainTextResponse(mesh.export(file_type='obj',include_texture=False))


@app.post('/api/runs/{ident}/cancel')
def cancel(ident:str):
    state=get_run(ident)
    if state['status'] in {'queued','running'}:(run_dir(ident)/'cancel').touch()
    return {'ok':True}


@app.get('/api/files/{relative:path}')
def files(relative:str):
    p=(ROOT/relative).resolve()
    allowed=[(ROOT/'reference_images').resolve(),RUNS.resolve()]
    if not any(p.is_relative_to(a) for a in allowed) or not p.is_file():raise HTTPException(404,'Файл не найден')
    # No local tokens, config files or executable HTML can be served.
    if p.suffix.lower() not in {'.png','.jpg','.jpeg','.webp','.obj','.glb','.json','.txt','.log','.csv','.npz','.ply'}:raise HTTPException(403,'Этот тип файла не доступен')
    return FileResponse(p,headers={'X-Content-Type-Options':'nosniff'})


@app.post('/api/uploads')
async def upload(files:list[UploadFile]=File(...)):
    if not 2<=len(files)<=12:raise HTTPException(400,'Загрузите от 2 до 12 ракурсов одного объекта')
    prepared=[]
    for f in files:
        data=await f.read(8*1024*1024+1)
        if len(data)>8*1024*1024:raise HTTPException(400,'Максимум 8 МБ на изображение')
        try:
            im=Image.open(io.BytesIO(data))
            if im.width*im.height>24_000_000:raise ValueError('Слишком большое изображение')
            fmt=im.format;im.verify()
            suffix={'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}.get(fmt)
            if suffix is None:raise ValueError('Только PNG, JPEG и WebP')
        except Exception as e:raise HTTPException(400,'Некорректное изображение: '+str(e))
        prepared.append((data,suffix))
    folder=ROOT/'reference_images/uploads'/('upload_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
    for i,(data,suffix) in enumerate(prepared):(folder/f'view_{i:02d}{suffix}').write_bytes(data)
    return {'dataset_id':folder.relative_to(ROOT).as_posix()}





@app.get('/api/runs/{ident}/translations')
def get_run_translations(ident:str):
    from gai.localization import text_pairs
    try:
        state=read_run(ident);path=run_dir(ident)/'translations/status.json'
        return json.loads(path.read_text()) if path.is_file() else {'status':'missing','pairs':text_pairs(state)}
    except (OSError,ValueError):raise HTTPException(404,'Run not found')


@app.post('/api/runs/{ident}/translations')
def translate_run_text(ident:str):
    from gai.localization import translate_saved_run
    try:
        state=read_run(ident);out=run_dir(ident)/'translations';path=out/'status.json'
        if path.is_file() and json.loads(path.read_text())['status'] in {'queued','running','completed'}:return json.loads(path.read_text())
        if state['status'] in {'running','queued'}:raise HTTPException(409,'Translation is available when the experiment finishes')
        result={'status':'queued','pairs':{}};save_json(path,result)
        executor.submit(translate_saved_run,ident)
        return result
    except (OSError,ValueError):raise HTTPException(404,'Run not found')

app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='web')
