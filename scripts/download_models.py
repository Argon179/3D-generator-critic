"""Run with .venvs/qwen/bin/python. Credentials are read by huggingface_hub."""
import os
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
cfg=json.loads((ROOT/'configs/project.json').read_text())
for model,revision,patterns in [(cfg['sf3d_model'],cfg['sf3d_revision'],['config.yaml','model.safetensors']),(cfg['qwen_model'],cfg['qwen_revision'],['*.json','*.safetensors','*.txt','*.jinja'])]:
    print('Downloading',model,flush=True)
    snapshot_download(model,revision=revision,allow_patterns=patterns,max_workers=1)
