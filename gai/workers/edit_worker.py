import json
import os
import sys
import time
from pathlib import Path


def main():
    job=json.loads(Path(sys.argv[1]).read_text())
    sys.path.insert(0,job['repo'])
    import clip
    original_load=clip.load
    def local_clip_load(*args,**kwargs):
        kwargs.setdefault('download_root',str(Path(os.environ['TORCH_HOME']).parent/'clip'))
        return original_load(*args,**kwargs)
    clip.load=local_clip_load
    import yaml
    import torch
    import random
    import numpy as np
    import igl
    # Upstream expects libigl's old (barycentric, face_ids) return signature;
    # libigl 2.5 additionally returns sampled positions.
    original_sampler=igl.random_points_on_mesh
    def legacy_sampler(*args,**kwargs):
        result=original_sampler(*args,**kwargs)
        return result[0],result[1]
    igl.random_points_on_mesh=legacy_sampler
    from loop import loop
    if not torch.cuda.is_available():raise RuntimeError('CUDA недоступна для TextDeformer')
    source_tokens=clip.tokenize(job['base_description'],truncate=False)
    target_tokens=clip.tokenize(job['target_description'],truncate=False)
    if torch.equal(source_tokens,target_tokens):
        raise ValueError('Исходное и целевое описание совпадают для CLIP; деформация не запущена')
    cfg=yaml.safe_load((Path(job['repo'])/'example_config.yml').read_text())
    cfg.update(mesh=job['mesh'],output_path=job['out'],text_prompt=job['target_description'],base_text_prompt=job['base_description'],epochs=job['epochs'],batch_size=job['batch_size'],train_res=job['train_res'],log_interval_im=max(10,job['epochs']//3),seed=42)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    # Upstream's import-time loader calls compile CUDA kernels here.
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    loop(cfg)
    (Path(job['out'])/'runtime.json').write_text(json.dumps({'elapsed_seconds':time.monotonic()-started,'peak_vram_mb':torch.cuda.max_memory_allocated()/1024**2,'epochs':job['epochs']}))


if __name__=='__main__':main()
