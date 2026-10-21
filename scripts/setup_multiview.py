"""Install isolated shape-only Hunyuan and DA3 environments; preserve existing stack."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gai.core import child_env


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--component',choices=['hunyuan','da3','all'],default='all')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    import torch
    if not torch.__version__.startswith('2.5.1'):
        raise SystemExit('Запускайте Python с torch 2.5.1, например .venvs/sf3d/bin/python')
    components=['hunyuan','da3'] if args.component=='all' else [args.component]
    lock={}
    cfg=json.loads((ROOT/'configs/project.json').read_text())
    for name in components:
        repo=ROOT/'vendor'/('Hunyuan3D-2' if name=='hunyuan' else 'Depth-Anything-3')
        if not (repo/'.git').exists():
            url='https://github.com/'+('Tencent-Hunyuan/Hunyuan3D-2' if name=='hunyuan' else 'ByteDance-Seed/Depth-Anything-3')+'.git'
            subprocess.run(['git','clone','--depth','1',url,str(repo)],check=True)
        lock[name]=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        folder=ROOT/'.venvs'/name
        if not (folder/'bin/python').exists():venv.create(folder,with_pip=True,system_site_packages=True)
        py=str(folder/'bin/python');env=child_env();env['PATH']=str(folder/'bin')+os.pathsep+env['PATH']
        common=['numpy==1.26.4','Pillow>=10,<12','trimesh>=4.6,<5','scipy==1.13.1','huggingface_hub>=0.34,<1','safetensors','einops','omegaconf','opencv-python-headless==4.10.0.84','rembg==2.0.61','onnxruntime','scikit-image==0.25.2']
        specific=['diffusers==0.32.2','transformers==4.46.3','accelerate==1.11.0','pymeshlab==2023.12.post3','pygltflib'] if name=='hunyuan' else ['open3d==0.19.0','evo','e3nn','moviepy==1.0.3','plyfile','pillow_heif','pycolmap','typer>=0.9','imageio','loguru']
        subprocess.run([py,'-m','pip','install','--find-links',str(ROOT/'.cache/downloads'),'-c',str(ROOT/'configs/torch-constraints.txt'),*common,*specific],env=env,check=True)
        if name=='da3':subprocess.run([py,'-m','pip','install','--no-deps','-e',str(repo)],env=env,check=True)
        probe=("import sys;sys.path.insert(0,"+repr(str(repo))+");from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline" if name=='hunyuan' else "from depth_anything_3.api import DepthAnything3;import open3d")
        subprocess.run([py,'-c',probe],env=env,check=True)
        (folder/'ready.json').write_text(json.dumps({'source_commit':lock[name],'imports_checked':True})+'\n')
        if args.download:
            model=cfg[name+'_model']
            patterns=['hunyuan3d-dit-v2-mv/config.yaml','hunyuan3d-dit-v2-mv/model.fp16.safetensors'] if name=='hunyuan' else ['config.json','model.safetensors']
            code='from huggingface_hub import snapshot_download; snapshot_download('+repr(model)+',revision='+repr(cfg.get(name+'_revision'))+',allow_patterns='+repr(patterns)+',max_workers=1)'
            subprocess.run([py,'-c',code],env=env,check=True)
    path=ROOT/'configs/multiview-upstream-lock.json'
    old=json.loads(path.read_text()) if path.exists() else {}
    old.update(lock);path.write_text(json.dumps(old,indent=2)+'\n')


if __name__=='__main__':main()
