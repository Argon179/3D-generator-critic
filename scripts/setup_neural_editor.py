"""Install optional SDS dependencies and download only pinned FP16 components.
Run with .venvs/textdeformer/bin/python scripts/setup_neural_editor.py.
"""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
    subprocess.check_call([sys.executable,'-m','pip','install','diffusers==0.32.2','transformers==4.46.3','accelerate==1.2.1','huggingface-hub<1','pydantic>=2.10,<3'])
    # nvdiffrast is provided by the project's existing TextDeformer environment.
    import nvdiffrast.torch
    from huggingface_hub import snapshot_download
    cfg=json.loads((ROOT/'configs/project.json').read_text())
    path=snapshot_download(cfg['neural_editor_model'],revision=cfg['neural_editor_revision'],allow_patterns=[
        'model_index.json','scheduler/*.json','tokenizer/*','text_encoder/config.json',
        'text_encoder/model.fp16.safetensors','unet/config.json','unet/diffusion_pytorch_model.fp16.safetensors',
        'vae/config.json','vae/diffusion_pytorch_model.fp16.safetensors'],max_workers=2)
    print('Neural editor ready:',path)

if __name__=='__main__':main()
