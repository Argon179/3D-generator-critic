"""English source text and Russian translations; never translate decision fields."""
import re

TEXT_FIELDS={'summary','observation','suggestion','reason','explanation','note','part'}


def bilingual_report(report,translate):
    """translate(texts, source, target) returns equally many translated strings.

    Only human-readable fields are touched. Booleans, operations, boxes, numbers,
    IDs and descriptions used by editors retain their exact original values.
    """
    import copy
    result=copy.deepcopy(report);warning_nodes=[];requests={'en':{},'ru':{}};nodes=[]
    def visit(value):
        if isinstance(value,list):
            for item in value:visit(item)
        if not isinstance(value,dict):return
        if isinstance(value.get('quality_warnings'),list):
            warnings=[{'reason_en':en,'reason_ru':ru} for en,ru in zip(value.get('quality_warnings_en') or [None]*len(value['quality_warnings']),value['quality_warnings'])]
            warning_nodes.append((value,warnings))
            for warning in warnings:visit(warning)
        for child in list(value.values()):visit(child)
        for field in TEXT_FIELDS:
            en=value.get(field+'_en');ru=value.get(field+'_ru')
            if field=='part' and not en and not ru:en=value.get('part')
            if not isinstance(en,str) and not isinstance(ru,str):continue
            if not en:
                if re.search('[А-Яа-яЁё]',ru or ''):requests['en'].setdefault(ru,None)
                else:en=ru
            if en:
                value[field+'_en']=en
                if not ru or not re.search('[А-Яа-яЁё]',ru):requests['ru'].setdefault(en,None)
            nodes.append((value,field,en,ru))
    visit(result)
    for target in ['en','ru']:
        texts=list(requests[target])
        for start in range(0,len(texts),4):
            batch=texts[start:start+4]
            translated=translate(batch,'ru' if target=='en' else 'en',target)
            if len(translated)!=len(batch) or not all(isinstance(t,str) and t.strip() for t in translated):raise ValueError('Incomplete translation batch')
            requests[target].update(zip(batch,translated))
    for node,field,en,ru in nodes:
        if not en:en=requests['en'][ru]
        node[field+'_en']=en
        node[field+'_ru']=ru if ru and re.search('[А-Яа-яЁё]',ru) else requests['ru'][en]
    for node,warnings in warning_nodes:
        node['quality_warnings']=[w['reason_ru'] for w in warnings]
        node['quality_warnings_en']=[w['reason_en'] for w in warnings]
    result['text_language']='en';result['translation_language']='ru'
    return result


def text_pairs(value):
    result={}
    def visit(node):
        if isinstance(node,list):
            for item in node:visit(item)
        if not isinstance(node,dict):return
        for ru,en in zip(node.get('quality_warnings',[]),node.get('quality_warnings_en',[])):
            if isinstance(ru,str) and isinstance(en,str):result[ru]=en
        for key,text in node.items():
            if key.endswith('_ru') and isinstance(text,str) and isinstance(node.get(key[:-3]+'_en'),str):result[text]=node[key[:-3]+'_en']
            visit(text)
    visit(value)
    return result


def translate_saved_run(ident):
    """Optional presentation-only translation of legacy run text, under the GPU lock."""
    import json,fcntl,subprocess
    from gai.core import ROOT,RUNS,config,read_run,save_json,child_env,local_path
    out=RUNS/ident/'translations';out.mkdir(exist_ok=True)
    try:
        cfg=config();state=read_run(ident)
        # Only text-bearing assessment objects, never regenerate or change acceptance.
        report={'iterations':[{'number':i['number'],'critique':i['critique'],'verification':i.get('verification')} for i in state.get('iterations',[])]}
        job={'model':cfg.get('qwen_review_model',cfg['qwen_model']),'revision':cfg.get('qwen_review_revision',cfg.get('qwen_revision')),
             'mode':'translate','max_pixels':65536,'report':report,'out':str(out)}
        save_json(out/'job.json',job)
        with (RUNS/'.gpu.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with (out/'process.log').open('w') as log:
                subprocess.run([str(local_path(cfg['qwen_python'])),str(ROOT/'gai/workers/qwen_worker.py'),str(out/'job.json')],env=child_env(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        localized=json.loads((out/'localized_text.json').read_text())
        save_json(out/'status.json',{'status':'completed','pairs':text_pairs(localized),'legacy_translation':True})
    except Exception as exc:
        save_json(out/'status.json',{'status':'failed','error':str(exc)})
