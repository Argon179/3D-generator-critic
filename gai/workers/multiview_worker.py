"""Isolated Hunyuan/DA3 inference; no model imports in the HTTP process."""
import json
import sys
import time
from pathlib import Path


def prepare(job):
    from PIL import Image
    from gai.masks import uniform_background
    out = Path(job['out']); out.mkdir(parents=True, exist_ok=True)
    refs = []
    session = None
    for n, ref in enumerate(job['references']):
        im = Image.open(ref['path']).convert('RGBA')
        if ref.get('crop'):
            x, y, w, h = ref['crop']
            im = im.crop((int(x*im.width), int(y*im.height), int((x+w)*im.width), int((y+h)*im.height)))
        im.thumbnail((1024, 1024))
        if job['background'] in {'black', 'white'}:
            im = uniform_background(im)
        elif job['background'] == 'rembg':
            import rembg
            session = session or rembg.new_session('u2net', providers=['CPUExecutionProvider'])
            im = rembg.remove(im, session=session)
        if im.getextrema()[3] == (255, 255):
            raise ValueError('Нужен прозрачный фон или удаление фона')
        box = im.getchannel('A').getbbox()
        if box is None:
            raise ValueError('После удаления фона объект не найден')
        im = im.crop(box)
        side = max(im.size)
        canvas = Image.new('RGBA', (round(side/.85), round(side/.85)))
        canvas.alpha_composite(im, ((canvas.width-im.width)//2, (canvas.height-im.height)//2))
        dest = out / f'reference_{n:02d}.png'; canvas.save(dest)
        refs.append(dict(ref, prepared=str(dest)))
    return refs


def hunyuan(job, refs):
    import torch
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    import os
    from gai.multiview import cached_snapshot
    snapshot = cached_snapshot(os.environ['HF_HOME'], job['model'], [
        job['subfolder']+'/config.yaml', job['subfolder']+'/model.fp16.safetensors'],job.get('revision'))
    if snapshot is None:raise FileNotFoundError('Локальные веса Hunyuan не найдены')
    job['resolved_revision'] = snapshot.name
    # Map the checkpoint into memory and bind tensors to empty modules. Avoid
    # temporarily duplicating ~5 GB of weights on a 16 GB host.
    import yaml
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from hy3dgen.shapegen.pipelines import instantiate_from_config
    config = yaml.safe_load((snapshot/job['subfolder']/'config.yaml').read_text())
    checkpoint = load_file(str(snapshot/job['subfolder']/'model.fp16.safetensors'), device='cpu')
    states = {}
    for key, value in checkpoint.items():
        component, name = key.split('.', 1)
        states.setdefault(component, {})[name] = value
    with init_empty_weights():
        modules = {name: instantiate_from_config(config[name]) for name in ('model','vae','conditioner')}
    # Released shape checkpoints contain only the VAE decoder; the encoder
    # and posterior projection are used for training/mesh encoding, never generation.
    modules['vae'].encoder = None
    modules['vae'].pre_kl = None
    for name, module in modules.items():
        result = module.load_state_dict(states[name], strict=False, assign=True)
        if result.missing_keys:
            raise RuntimeError(f'Checkpoint {name}: missing weights: {result.missing_keys[:5]}')
        module.eval()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline(**modules,
        image_processor=instantiate_from_config(config['image_processor']),
        scheduler=instantiate_from_config(config['scheduler']), device='cpu', dtype=torch.float16)
    del checkpoint, states
    # Upstream offload expects the Diffusers-style components mapping, which
    # its standalone constructor does not populate.
    pipeline.components = modules
    pipeline.enable_model_cpu_offload(device='cuda')
    # This upstream pipeline uses self.device for sampling even with offload hooks.
    pipeline.device = torch.device('cuda')
    images = {side: Image.open(refs[index]['prepared']) for side, index in job['selected'].items()}
    mesh = pipeline(image=images, num_inference_steps=job.get('steps', 30),
                    octree_resolution=job.get('resolution', 256), num_chunks=8000,
                    generator=torch.Generator(device='cuda').manual_seed(job['seed']), output_type='trimesh')[0]
    if mesh is None:
        raise RuntimeError('Hunyuan вернул пустую поверхность')
    return mesh


def da3(job, refs):
    import numpy as np
    import torch
    import open3d as o3d
    from PIL import Image
    from depth_anything_3.api import DepthAnything3
    selected = job['selected']
    images = []
    for index in selected:
        im = Image.open(refs[index]['prepared']).convert('RGBA')
        background = Image.new('RGBA', im.size, (127,127,127,255))
        images.append(Image.alpha_composite(background, im).convert('RGB'))
    import os
    from gai.multiview import cached_snapshot
    snapshot=cached_snapshot(os.environ['HF_HOME'],job['model'],['config.json','model.safetensors'],job.get('revision'))
    if snapshot is None:raise FileNotFoundError('Локальные веса DA3 не найдены')
    job['resolved_revision'] = snapshot.name
    model = DepthAnything3.from_pretrained(str(snapshot)).to('cuda').eval()
    prediction = model.inference(images, process_res=job.get('resolution', 336), process_res_method='upper_bound_resize')
    del model
    torch.cuda.empty_cache()
    out = Path(job['out'])
    np.savez_compressed(out/'depth_cameras.npz', depth=prediction.depth, confidence=prediction.conf,
                        extrinsics=prediction.extrinsics, intrinsics=prediction.intrinsics,
                        selected=np.array(selected))
    # Fuse only foreground points. DA3 returns OpenCV world-to-camera matrices.
    points, colors = [], []
    for n, index in enumerate(selected):
        depth = prediction.depth[n]
        h, w = depth.shape
        rgba = Image.open(refs[index]['prepared']).convert('RGBA').resize((w,h))
        mask = np.array(rgba.getchannel('A')) > 220
        confidence = prediction.conf[n]
        finite = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & mask
        if finite.sum() < 20:
            continue
        finite &= confidence >= np.percentile(confidence[finite], 25)
        y, x = np.where(finite)
        z = depth[y,x]
        k = prediction.intrinsics[n]
        camera = np.column_stack(((x-k[0,2])*z/k[0,0], (y-k[1,2])*z/k[1,1], z))
        ext = prediction.extrinsics[n]
        world = (camera-ext[:3,3]) @ ext[:3,:3]
        points.append(world); colors.append(np.asarray(rgba)[y,x,:3]/255.)
        lo, hi = np.percentile(depth[finite], [2,98])
        visual = np.uint8(np.clip((depth-lo)/max(hi-lo,1e-6),0,1)*255)
        Image.fromarray(visual).save(out/f'depth_{index:02d}.png')
    if not points:
        raise ValueError('DA3: нет уверенных точек объекта')
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(points))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colors))
    span = np.ptp(np.asarray(cloud.points), axis=0).max()
    if not np.isfinite(span) or span <= 0:
        raise ValueError('DA3: вырожденное облако точек')
    cloud = cloud.voxel_down_sample(span/200)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.)
    if len(cloud.points) < 100:
        raise ValueError('DA3: недостаточно точек для сетки')
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=span/30, max_nn=40))
    cloud.orient_normals_consistent_tangent_plane(30)
    o3d.io.write_point_cloud(str(out/'points.ply'), cloud)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=8, n_threads=2)
    mesh.remove_vertices_by_mask(np.asarray(density) < np.quantile(density,.02))
    mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
    import trimesh
    result = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles), process=True)
    # Align the estimated world frame to the first input camera, then Y-up.
    first = prediction.extrinsics[0]
    result.vertices = result.vertices @ first[:3,:3].T + first[:3,3]
    result.vertices[:,1:] *= -1
    (out/'reconstruction_notes.txt').write_text('DA3 predicted cameras and depth; foreground fusion + Poisson. Not metric CAD. Unseen regions may be filled by Poisson; topology and scale need validation.')
    return result


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    sys.path.insert(0, job['repo'])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна')
    started = time.monotonic()
    torch.manual_seed(job['seed']); torch.cuda.reset_peak_memory_stats()
    refs = prepare(job)
    selected = set(job['selected'].values() if isinstance(job['selected'],dict) else job['selected'])
    refs = [dict(r,used_for_generation=i in selected) for i,r in enumerate(refs)]
    out = Path(job['out'])
    (out/'references.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2))
    mesh = hunyuan(job,refs) if job['backend']=='hunyuan' else da3(job,refs)
    if len(mesh.faces)<4:
        raise ValueError('Реконструкция не создала пригодную сетку')
    mesh.export(out/'generated.glb')
    import resource
    (out/'memory.json').write_text(json.dumps({'peak_vram_mb':torch.cuda.max_memory_allocated()/1024**2,
        'peak_ram_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'seconds':time.monotonic()-started,'backend':job['backend']}))
    (out/'generation_quality.json').write_text(json.dumps({'backend':job['backend'],'selected':job['selected'],
        'heldout':[i for i in range(len(refs)) if i not in selected],'candidates':[],
        'model':job['model'],'revision':job.get('resolved_revision'),'note':'Holdout images were not passed to the neural model.'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
