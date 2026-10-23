"""VLM side suggestions, validated before they reach the form."""
import fcntl
import json
import subprocess
from pathlib import Path
from gai.core import ROOT,RUNS,child_env,config,local_path,save_json


def validate_labels(data,indexes):
    sides=data.get('sides',{})
    if not isinstance(sides,dict) or set(sides)-{'front','left','back','right'}:
        raise ValueError('VLM вернула неизвестные стороны')
    result={}
    for side,index in sides.items():
        if index is None:continue
        if type(index) is not int or index not in indexes:raise ValueError('VLM указала неизвестное фото')
        if index in result.values():raise ValueError('VLM назначила одному фото несколько сторон')
        result[side]=index
    if len(result)>=len(indexes):raise ValueError('VLM не оставила фото для проверки; назначьте стороны вручную')
    return {'sides':result,'note':str(data.get('note_ru','Проверьте предложенные стороны вручную.'))[:1000]}


def infer_labels(ident,refs):
    cfg=config();folder=ROOT/'.cache/annotations'/ident;folder.mkdir(parents=True,exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    with (RUNS/'.gpu.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('GPU занята генерацией. Повторите авторазметку после её завершения.')
        job={'model':cfg['qwen_model'],'revision':cfg.get('qwen_revision'),'references':refs,'out':str(folder)}
        save_json(folder/'job.json',job)
        with (folder/'process.log').open('w') as log:
            r=subprocess.run([str(local_path(cfg['qwen_python'])),str(ROOT/'gai/workers/label_worker.py'),str(folder/'job.json')],env=child_env(),stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if r.returncode:raise RuntimeError('Ошибка VLM. Подробности сохранены в .cache/annotations/'+ident+'/process.log')
        return validate_labels(json.loads((folder/'labels.json').read_text()),[r['index'] for r in refs])
