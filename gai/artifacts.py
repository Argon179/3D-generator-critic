"""Read-only stage inventory for both existing and newly created runs."""
import json
from pathlib import Path
from gai.core import ROOT,run_dir,read_run


def stage_inventory(ident):
    folder=run_dir(ident);state=read_run(ident)
    stages={}
    def stage(key,label):
        return stages.setdefault(key,{'id':key,'label':label,'models':[],'images':[],'files':[],'details':{}})
    inputs=stage('inputs','Входные фотографии и параметры');inputs['details']={'parameters':state['parameters'],'config_snapshot':state.get('config'),'references':state['references'],'created_at':state.get('created_at')}
    for p in sorted((folder/'source_images').glob('*')):
        ref=next((r for r in state['references'] if Path(r['path'])==p),{})
        inputs['images'].append({'path':str(p.relative_to(ROOT)),'label':p.name+(f" · {ref['azimuth_deg']}°" if ref.get('azimuth_deg') is not None else '')})
    generation=stage('generation',{'hunyuan':'Hunyuan3D-2mv · генерация','da3':'DA3 · глубина, камеры и сетка'}.get(state['parameters'].get('generator'),'SF3D · все кандидаты'))
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or p.is_symlink() or 'source_images' in p.parts or 'previews' in p.parts:continue
        rel=p.relative_to(folder);parts=rel.parts
        if len(parts)==1:
            if p.name in {'generation_job.json','process.log','error.txt'}:target=generation
            else:continue
        elif parts[0].startswith('iteration_'):
            number=int(parts[0].split('_')[1]);name=p.name
            if 'sf3d_evaluation' in parts:target=generation
            elif 'reconstruction' in parts:target=stage('reconstruction','Многовидовая реконструкция')
            elif 'shape_evidence' in parts:target=stage(f'critic_{number}',f'Критик и области формы · итерация {number}')
            elif 'verification' in parts:target=stage(f'verify_{number}',f'Независимая проверка до/после · итерация {number}')
            elif 'renders' in parts or name in {'render_job.json','metrics.json'}:
                target=stage(f'render_{number}',f'Рендеры и метрики · итерация {number}')
            elif name in {'critique.json','critic_job.json','raw_response.txt','critic_prompt.txt','view_assessments.json','proposal_choices.json','evaluated_mesh.json'} or name.startswith(('view_','plan_','choice_')) and p.suffix in {'.json','.txt'}:
                target=stage(f'critic_{number}',f'Критик · итерация {number}')
            elif number>0:target=stage(f'edit_{number}',f'Редактирование · итерация {number}')
            else:target=generation
        else:continue
        item={'path':str(p.relative_to(ROOT)),'label':str(rel),'size':p.stat().st_size}
        if p.suffix.lower() in {'.obj','.glb'}:
            target['models'].append(item)
        elif p.suffix.lower() in {'.png','.jpg','.jpeg'}:
            target['images'].append(item)
        elif p.suffix.lower() in {'.json','.txt','.log','.npz','.ply'}:
            target['files'].append(item)
            if p.suffix=='.json' and p.stat().st_size<200_000:
                try:
                    data=json.loads(p.read_text());target['details'][str(rel)]=data
                    if p.name=='comparison.json':target['comparison']=data
                except (OSError,ValueError):pass
    for item in state.get('iterations',[]):
        if item.get('backend')=='Многовидовая реконструкция по силуэтам' and 'reconstruction' in stages:
            stages['reconstruction']['comparison']=item
        if item['number']==0 and not (folder/'iteration_00/reconstruction').exists():generation['comparison']=item
        for key in (f'render_{item["number"]}',f'critic_{item["number"]}',f'verify_{item["number"]}'):
            target=stages.get(key)
            if target:
                model=ROOT/item['mesh']
                if model.is_file():target['models'].insert(0,{'path':item['mesh'],'label':'Геометрия этой итерации'})
                target['comparison']=item
    for number,item in enumerate(state.get('iterations',[])):
        target=stages.get(f"edit_{item['number']}")
        if target and number:
            target['models'].insert(0,{'path':item.get('edit_source_mesh',state['iterations'][number-1]['mesh']),'label':'До правки'})
            target['comparison']=item
    log=folder/'process.log'
    if log.exists():
        for target in stages.values():
            if not any(f['path']==str(log.relative_to(ROOT)) for f in target['files']):
                target['files'].append({'path':str(log.relative_to(ROOT)),'label':'Полный журнал всех этапов','size':log.stat().st_size})
    def order(s):
        key=s['id']
        if key in {'inputs','generation','reconstruction'}:return {'inputs':0,'generation':10,'reconstruction':20}[key]
        kind,number=key.rsplit('_',1);number=int(number)
        return {'render':30+number*30,'critic':40+number*30,'edit':15+number*30,'verify':45+number*30}[kind]
    return {'stages':sorted(stages.values(),key=order),'events':state['events']}
