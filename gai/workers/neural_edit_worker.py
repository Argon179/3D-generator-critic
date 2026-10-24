"""Experimental photo-anchored mesh deformation using a frozen SD 1.5 prior.

Input is the actual mesh, VLM-selected region, training photos and English text.
This is a custom SDS optimizer, not a pretrained mesh-to-mesh repair network.
Unselected vertices and connectivity are preserved; independent review is mandatory.
"""
import json
import sys
import time
from pathlib import Path


def run(job):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import trimesh
    import nvdiffrast.torch as dr
    from PIL import Image
    from diffusers import AutoencoderKL,UNet2DConditionModel,DDPMScheduler
    from transformers import CLIPTokenizer,CLIPTextModel
    from gai.neural_edit import prepare_neural_edit,deformation_weights
    from gai.shape_fit import mask_score
    from gai.local_editor import UnsupportedEdit
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for neural mesh editing')
    device='cuda';dtype=torch.float16
    torch.manual_seed(job.get('seed',42));torch.cuda.manual_seed_all(job.get('seed',42))
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    catalog,proposal,records=prepare_neural_edit(job['mesh'],job['source'],job['proposal_id'])
    mesh=trimesh.load(job['mesh'],force='mesh',process=False);original=mesh.vertices.copy()
    weights=deformation_weights(mesh,proposal['vertex_indexes'])
    base=torch.tensor(original,dtype=torch.float32,device=device)
    faces=torch.tensor(mesh.faces,dtype=torch.int32,device=device)
    edges=torch.tensor(mesh.edges_unique,dtype=torch.long,device=device)
    weight=torch.tensor(weights,device=device)[:,None]
    span=float(np.ptp(original,axis=0).max());limit=.04*span
    delta=torch.nn.Parameter(torch.zeros_like(base));optim=torch.optim.Adam([delta],lr=job.get('lr',.01))
    # All learned components are frozen. Only existing mesh vertices receive gradients.
    kw=dict(revision=job['revision'],local_files_only=True,use_safetensors=True,variant='fp16')
    tokenizer=CLIPTokenizer.from_pretrained(job['model'],subfolder='tokenizer',revision=job['revision'],local_files_only=True)
    text_model=CLIPTextModel.from_pretrained(job['model'],subfolder='text_encoder',torch_dtype=dtype,**kw).to(device).eval().requires_grad_(False)
    prompt='a detailed 3D sculpture of '+job['description']+'. '+job.get('instruction','')
    tokens=tokenizer(['',prompt],padding='max_length',max_length=77,truncation=True,return_tensors='pt').input_ids.to(device)
    with torch.no_grad():text=text_model(tokens)[0]
    effective_prompt=tokenizer.decode(tokens[1],skip_special_tokens=True)
    del text_model
    vae=AutoencoderKL.from_pretrained(job['model'],subfolder='vae',torch_dtype=dtype,**kw).to(device).eval().requires_grad_(False)
    unet=UNet2DConditionModel.from_pretrained(job['model'],subfolder='unet',torch_dtype=dtype,**kw).to(device).eval().requires_grad_(False)
    # Batch one view and one CFG pass at a time bounds activation memory on 8 GB.
    unet.set_attention_slice('auto');vae.enable_slicing()
    scheduler=DDPMScheduler.from_pretrained(job['model'],subfolder='scheduler',revision=job['revision'],local_files_only=True)
    alphas=scheduler.alphas_cumprod.to(device)
    ctx=dr.RasterizeCudaContext() if hasattr(dr,'RasterizeCudaContext') else dr.RasterizeGLContext();res=int(job.get('resolution',256))
    targets=[];matrices=[]
    for record in records:
        im=Image.open(record['target']).convert('RGBA').resize((res,res),Image.Resampling.LANCZOS)
        a=np.asarray(im).copy();targets.append(torch.tensor(a[:,:,3]/255,dtype=torch.float32,device=device)[None,:,:,None])
        matrices.append(torch.tensor(record['projection_matrix'],dtype=torch.float32,device=device))
    def render(vertices,matrix):
        pos=torch.cat([vertices,torch.ones_like(vertices[:,:1])],dim=1)@matrix.T
        rast,_=dr.rasterize(ctx,pos[None].contiguous(),faces,resolution=[res,res])
        tri=vertices[faces.long()];normal=torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1)
        vn=torch.zeros_like(vertices)
        for k in range(3):vn=vn.index_add(0,faces[:,k].long(),normal)
        vn=F.normalize(vn,dim=-1,eps=1e-6)
        light=torch.tensor([.3,.7,.6],device=device)
        shade=.35+.6*(vn@light).abs().clamp(0,1)
        color=shade[:,None].expand(-1,3).contiguous()
        rgb,_=dr.interpolate(color[None],rast,faces)
        mask=(rast[:,:,:,3:]>0).float()
        rgba=torch.cat([rgb*mask+(1-mask),mask],dim=-1)
        rgba=dr.antialias(rgba.contiguous(),rast,pos[None].contiguous(),faces)
        return rgba.flip(1)
    reference_normals=torch.tensor(mesh.face_normals.copy(),device=device,dtype=torch.float32)
    reliable=torch.tensor(mesh.area_faces>max(mesh.area*1e-12,1e-14),device=device)
    before=mask_score(mesh,catalog['views']);history=[]
    best=original.copy();last_legal=original.copy()
    with torch.no_grad():best_loss=float(sum(F.mse_loss(render(base,m)[:,:,:,3:],target) for m,target in zip(matrices,targets))/len(records))
    for step in range(int(job.get('steps',60))):
        optim.zero_grad(set_to_none=True)
        vertices=base+torch.tanh(delta)*limit*weight
        index=step%len(records);rgba=render(vertices,matrices[index]);rgb=rgba[:,:,:,:3].permute(0,3,1,2)
        # SDS: frozen diffusion gives a text-conditioned image gradient; VAE and
        # differentiable rasterizer propagate it back to mesh coordinates.
        latents=vae.encode((rgb*2-1).to(dtype)).latent_dist.mode()*vae.config.scaling_factor
        timestep=torch.randint(50,500,(1,),device=device,dtype=torch.long)
        with torch.no_grad():
            noise=torch.randn_like(latents);noisy=scheduler.add_noise(latents.detach(),noise,timestep)
            eps0=unet(noisy,timestep,encoder_hidden_states=text[:1]).sample
            eps1=unet(noisy,timestep,encoder_hidden_states=text[1:]).sample
            eps=eps0+7.5*(eps1-eps0)
            grad=((1-alphas[timestep])*(eps-noise)).float()
            grad=torch.nan_to_num(grad,nan=0,posinf=0,neginf=0).clamp(-1,1)
        sds=(latents.float()*grad.detach()).sum()/latents.numel()
        # Train against ALL selected photos each step, never against holdout views.
        photo=0
        for j,matrix in enumerate(matrices):
            alpha=rgba[:,:,:,3:] if j==index else render(vertices,matrix)[:,:,:,3:]
            photo=photo+F.mse_loss(alpha,targets[j])/len(records)
        displacement=vertices-base
        smooth=((displacement[edges[:,0]]-displacement[edges[:,1]])/span).square().mean()
        anchor=(displacement/span).square().mean()
        loss=photo*20+sds*.05+smooth*10+anchor
        if not torch.isfinite(loss):raise UnsupportedEdit('Nonfinite neural editing loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([delta],1);optim.step()
        # Reject folded/degenerate candidates and back off instead of exporting tears.
        with torch.no_grad():candidate=(base+torch.tanh(delta)*limit*weight).cpu().numpy()
        mesh.vertices=candidate
        legal=np.isfinite(candidate).all() and np.all(mesh.area_faces[reliable.cpu().numpy()]>1e-14)
        legal=legal and np.all((mesh.face_normals[reliable.cpu().numpy()]*reference_normals[reliable].cpu().numpy()).sum(1)>.1)
        if not legal:
            mesh.vertices=last_legal
            with torch.no_grad():
                ratio=torch.tensor((last_legal-original)/limit,device=device)/(weight+1e-8)
                delta.copy_(torch.atanh(ratio.clamp(-.999,.999)))
            optim.state.clear()
        else:last_legal=candidate.copy()
        # Select checkpoints by the training objective only; controls never tune it.
        with torch.no_grad():score=float(sum(F.mse_loss(render(torch.tensor(candidate,device=device,dtype=torch.float32),m)[:,:,:,3:],target) for m,target in zip(matrices,targets))/len(records))
        if legal and score<best_loss and step>0:best_loss=score;best=candidate.copy()
        if step%10==0 or step==job.get('steps',60)-1:
            history.append(dict(step=step+1,photo_loss=score,sds_surrogate=float(sds.detach()),legal=bool(legal)))
            print(json.dumps(history[-1]),flush=True)
    mesh.vertices=best
    displacement=np.linalg.norm(best-original,axis=1)
    if displacement.max()<1e-7:raise UnsupportedEdit('Neural editor produced no valid deformation')
    path=out/'mesh.obj';mesh.export(path)
    training=[r['reference_index'] for r in records]
    report={'backend':'Stable Diffusion 1.5 mesh SDS (experimental)','model':job['model'],'revision':job['revision'],
        'effective_prompt':effective_prompt,'instruction':job.get('instruction',''),'plan':{'operation':'neural_refine','proposal_id':proposal['id'],'regions':proposal['regions']},
        'training_views':training,'control_views':[r['reference_index'] for r in catalog['views'] if r['reference_index'] not in training and not r.get('used_for_generation',False)],
        'reserved_control':catalog.get('reserved_control'),'before_fixed_frame_iou':before,'after_fixed_frame_iou':mask_score(mesh,catalog['views']),
        'topology_preserved':True,'selected_vertices':int((weights>0).sum()),'max_displacement':float(displacement.max()),'history':history,
        'elapsed_seconds':time.monotonic()-started,'peak_vram_mb':torch.cuda.max_memory_allocated()/1024**2,
        'limitations':'Custom SDS mesh optimizer, not a pretrained mesh repair model. Fixed connectivity cannot add/remove parts. Approximate cameras and a 2D prior can cause artifacts; independent validation required.'}
    (out/'edit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));return path


if __name__=='__main__':
    job=json.loads(Path(sys.argv[1]).read_text())
    try:run(job)
    except Exception as exc:
        out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
        (out/'neural_error.json').write_text(json.dumps({'error':str(exc),'type':type(exc).__name__}))
        raise

