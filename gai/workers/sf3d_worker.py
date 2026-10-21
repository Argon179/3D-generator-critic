"""Run in the isolated SF3D environment. Never imported by the web process."""
import json
import sys
from pathlib import Path


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    sys.path.insert(0, job['repo'])
    import rembg
    import torch
    import numpy as np
    from PIL import Image
    from sf3d.system import SF3D
    from sf3d.utils import resize_foreground
    from huggingface_hub import snapshot_download
    from sf3d.models.tokenizers.dinov2 import Dinov2Model, Dinov2Config
    from unittest.mock import patch
    import open_clip
    from gai.masks import uniform_background
    from gai.geometry import geometry_report
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна. Запустите приложение вне песочницы.')
    torch.manual_seed(job['seed'])
    torch.cuda.manual_seed_all(job['seed'])
    out = Path(job['out'])
    out.mkdir(parents=True, exist_ok=True)
    session = None
    refs = []
    for n, ref in enumerate(job['references']):
        im = Image.open(ref['path']).convert('RGBA')
        if ref.get('crop'):
            x, y, w, h = ref['crop']
            im = im.crop((int(x*im.width), int(y*im.height), int((x+w)*im.width), int((y+h)*im.height)))
        im.thumbnail((1024,1024))
        if job['background'] in {'black','white'}:
            im=uniform_background(im)
        elif job['background'] == 'rembg':
            session = session or rembg.new_session('u2net', providers=['CPUExecutionProvider'])
            im = rembg.remove(im, session=session)
        if im.getextrema()[3] == (255,255):
            raise ValueError('Нужен прозрачный фон или режим удаления фона')
        # Same framing for generator and silhouette/VLM comparisons.
        im = resize_foreground(im, 0.85)
        dest = out / f'reference_{n:02d}.png'
        im.save(dest)
        refs.append(dict(ref,prepared=str(dest)))
    (out/'references.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2))
    snapshot = snapshot_download(job['model'],revision=job.get('revision'),allow_patterns=['config.yaml','model.safetensors'],max_workers=1)
    # The SF3D checkpoint already includes ALL DINO and OpenCLIP parameters.
    # Upstream otherwise downloads their original checkpoints before overwriting
    # those parameters. Construct architectures only; SF3D's strict safetensors
    # load still verifies and restores every parameter (no random-weight inference).
    def dino_architecture(cls,name,*args,**kwargs):
        return cls(Dinov2Config.from_pretrained(name))
    create_clip=open_clip.create_model_and_transforms
    def clip_architecture(*args,**kwargs):
        kwargs['pretrained']=None
        return create_clip(*args,**kwargs)
    with patch.object(Dinov2Model,'from_pretrained',classmethod(dino_architecture)), patch.object(open_clip,'create_model_and_transforms',clip_architecture):
        model = SF3D.from_pretrained(snapshot, config_name='config.yaml', weight_name='model.safetensors').to('cuda').eval()
    torch.cuda.reset_peak_memory_stats()
    candidates=[]
    def generate(index):
        with torch.inference_mode(), torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            mesh, _ = model.run_image(Image.open(refs[index]['prepared']),bake_resolution=512,remesh='none',vertex_count=-1)
        report=geometry_report(mesh)
        mesh.export(out/f'candidate_{index:02d}.glb',include_normals=True)
        candidates.append({'reference_index':index,'geometry':report})
        print(f"SF3D reference {index}: thickness ratio {report['thickness_ratio']:.3f}",flush=True)
        return mesh,report
    mesh,report=generate(0)
    chosen=0
    # SF3D is single-view. Retry individual available photos rather than stretch
    # the flattened mesh and pretend missing geometry has been reconstructed.
    if report['severity']!='none' and len(refs)>1:
        def priority(index,target):
            im=Image.open(refs[index]['prepared'])
            box=im.getchannel('A').getbbox()
            width,height=box[2]-box[0],box[3]-box[1]
            angle=refs[index].get('azimuth_deg')
            base=refs[0].get('azimuth_deg')
            separation=min((angle-base)%360,(base-angle)%360) if angle is not None and base is not None else 90
            return (-abs(min(separation,180-separation)-target),width/max(height,1))
        pool=list(range(1,len(refs)));remaining=[]
        for target in (90,45):
            if pool:
                index=max(pool,key=lambda i:priority(i,target));remaining.append(index);pool.remove(index)
        for index in remaining:
            candidate,candidate_report=generate(index)
            if candidate_report['thickness_ratio']>report['thickness_ratio']:
                mesh,report,chosen=candidate,candidate_report,index
            if report['severity']=='none':break
    mesh.export(out/'generated.glb',include_normals=True)
    # Exclude EVERY attempted generation photograph from holdout acceptance.
    attempted={c['reference_index'] for c in candidates}
    refs=[dict(r,used_for_generation=i in attempted) for i,r in enumerate(refs)]
    refs=[refs[chosen],*[r for i,r in enumerate(refs) if i!=chosen]]
    (out/'references.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2))
    (out/'generation_quality.json').write_text(json.dumps({'selected_reference_index':chosen,
        'candidates':candidates,'selection':'Largest thickness ratio among attempted views; not a 3D accuracy score.'},ensure_ascii=False,indent=2))
    (out/'memory.json').write_text(json.dumps({'peak_vram_mb':torch.cuda.max_memory_allocated()/1024**2}))


if __name__ == '__main__':
    main()
