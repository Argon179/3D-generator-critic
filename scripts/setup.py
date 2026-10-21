"""Recreate project environments; invoke with the Python that already has torch 2.5.1+cu121.

Example: /path/to/torch/python scripts/setup.py --component qwen
Does not install a GPU driver or modify the source Python environment.
"""
import argparse
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
    parser.add_argument('--component',choices=['base','sf3d','qwen','textdeformer','all'],default='all')
    args=parser.parse_args()
    import torch
    if torch.__version__!='2.5.1+cu121':
        raise SystemExit('Для этих requirements нужен Python с torch==2.5.1+cu121 и torchvision==0.20.1+cu121. Существующие окружения не изменены.')
    names=['base','sf3d','qwen','textdeformer'] if args.component=='all' else [args.component]
    for name in names:
        folder=ROOT/('.venv' if name=='base' else '.venvs/'+name)
        if not (folder/'bin/python').exists():venv.create(folder,with_pip=True,system_site_packages=True)
        py=str(folder/'bin/python');env=child_env();env['PATH']=str(folder/'bin')+os.pathsep+env['PATH']
        def pip(*items):subprocess.run([py,'-m','pip','install','--no-cache-dir',*items],env=env,cwd=ROOT,check=True)
        if name=='base':
            pip('-e','.', 'fast-simplification==0.1.12','pytest','httpx')
        elif name=='qwen':
            pip('-c','configs/torch-constraints.txt','transformers==4.57.1','accelerate==1.11.0','bitsandbytes==0.48.2','numpy==1.26.4','Pillow>=10,<12')
        elif name=='sf3d':
            pip('-c','configs/torch-constraints.txt','-r','configs/requirements-sf3d.txt')
            env.update(USE_CUDA='1',TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='2')
            pip('--no-build-isolation','./vendor/stable-fast-3d/texture_baker','./vendor/stable-fast-3d/uv_unwrapper')
        else:
            pip('-c','configs/torch-constraints.txt','numpy==1.26.4','scipy==1.13.1','libigl==2.5.1','pymeshlab==2023.12.post3','setuptools==75.8.2','wheel','cholespy','cython','imageio','imageio-ffmpeg','kornia','PyGLM','resize-right','smplx','tqdm','ninja','pyyaml','matplotlib','trimesh','tensorboard','easydict','fire','ftfy','regex')
            pip('--only-binary=:all:','torch-scatter','torch-sparse','-f','https://data.pyg.org/whl/torch-2.5.0+cu121.html')
            pip('--no-build-isolation','--no-deps','git+https://github.com/openai/CLIP.git@b46f5ac7587d2e1862f8b7b1573179d80dcdd620','git+https://github.com/NVlabs/nvdiffrast.git@78528e683210dfaa1be57e3c65aa37d3b36c6644')


if __name__=='__main__':main()
