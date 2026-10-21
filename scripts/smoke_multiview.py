"""Run a real local multiview generation through the normal pipeline, no fake models."""
import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gai.core import config,library,now,run_dir,save_json
from gai.pipeline import execute


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['hunyuan','da3'],required=True)
    parser.add_argument('--object',choices=['rubber_duck','lucky_cat','frog'],default='rubber_duck')
    args=parser.parse_args()
    dataset=next(d for d in library() if d['id'].endswith('/'+args.object))
    ident=uuid.uuid4().hex[:12];folder=run_dir(ident);(folder/'source_images').mkdir(parents=True)
    refs=[]
    generation_index=9 if args.object=='rubber_duck' else 0
    ordered=[generation_index]+[i for i in range(len(dataset['images'])) if i!=generation_index]
    for i,index in enumerate(ordered):
        ref=dataset['images'][index]
        dest=folder/'source_images'/f'view_{i:02d}.png';shutil.copy2(ROOT/ref['path'],dest)
        refs.append(dict(ref,path=str(dest),original_path=str(ROOT/ref['path']),crop=None))
    parameters={'dataset_id':dataset['id'],'image_indexes':list(range(len(refs))),'generation_index':generation_index,
        'generator':args.backend,'prompt':dataset['prompt'],'seed':42,'iterations':0,'background':'black','crop':None,'view_sides':None}
    state={'id':ident,'name':dataset['name'],'created_at':now(),'updated_at':now(),'status':'queued','stage':'queued',
        'parameters':parameters,'references':refs,'iterations':[],'events':[],'config':config()}
    save_json(folder/'state.json',state)
    print('RUN_ID='+ident,flush=True)
    execute(ident)
    result=json.loads((folder/'state.json').read_text())
    print(json.dumps({'id':ident,'status':result['status'],'error':result.get('error')},ensure_ascii=False),flush=True)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
