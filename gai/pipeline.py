from __future__ import annotations

import fcntl
import json
import math
import shutil
import sys
import traceback
import re
from pathlib import Path

import numpy as np
import trimesh

from gai.core import ROOT, RUNS, Cancelled, command, config, event, local_path, read_run, run_dir, save_json, update_run
from gai.metrics import align, acceptance, validation_summary
from gai.geometry import geometry_report, enforce_geometry_warning
from gai.critic_schema import Critique, EditPlan
from gai.local_editor import apply_plan, UnsupportedEdit


def canonical_mesh(source, destination, max_faces):
    mesh=trimesh.load(source,force='mesh',process=True)
    if not isinstance(mesh,trimesh.Trimesh) or len(mesh.faces)<4 or not np.isfinite(mesh.vertices).all():
        raise ValueError('Некорректная или пустая сетка')
    # GLB UV seams duplicate vertices. TextDeformer's Poisson solve needs actual
    # geometric connectivity, so weld geometry independently of UVs/normals.
    mesh.visual=trimesh.visual.ColorVisuals(mesh)
    mesh.merge_vertices(merge_tex=True,merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces)<4:raise ValueError('После удаления вырожденных граней сетка пуста')
    components=mesh.split(only_watertight=False)
    removed=0
    if len(components)>1:
        largest=max(components,key=lambda p:len(p.faces))
        if len(largest.faces)/len(mesh.faces)>=.98:
            removed=len(mesh.faces)-len(largest.faces)
            mesh=largest
    if len(mesh.faces)>max_faces:
        # Avoid a heavy sparse solve for a 100k-face SF3D result.
        try:mesh=mesh.simplify_quadric_decimation(face_count=max_faces)
        except ImportError as e:raise RuntimeError('Установите fast-simplification в базовое окружение') from e
    center=mesh.bounds.mean(axis=0)
    span=float(np.ptp(mesh.vertices,axis=0).max())
    if span<=1e-8:raise ValueError('Размер сетки равен нулю')
    scale=2/span
    mesh.vertices=(mesh.vertices-center)*scale
    mesh.export(destination)
    return {'vertices':len(mesh.vertices),'faces':len(mesh.faces),'source_center':center.tolist(),'source_scale':float(scale),'canonical_up':'Y','removed_fragment_faces':removed,'can_deform':mesh.body_count==1,'geometry':geometry_report(mesh),'materials':'clay comparison; source GLB retains original materials'}


def validate_candidate(path):
    mesh=trimesh.load(path,force='mesh',process=False)
    if not isinstance(mesh,trimesh.Trimesh) or not np.isfinite(mesh.vertices).all() or len(mesh.faces)<4:
        raise ValueError('Редактор вернул некорректную сетку')
    if np.ptp(mesh.vertices,axis=0).max()>6:
        raise ValueError('Сильная деформация: размер сетки вышел за допустимый предел')


def render(run_id, mesh, folder, cfg):
    job=folder/'render_job.json'
    save_json(job,{'mesh':str(mesh),'out':str(folder/'renders'),'resolution':cfg['render_resolution'],'step':cfg['render_step_deg'],'elevations':cfg['render_elevations'],'views':cfg.get('_render_views')})
    command(run_id,[local_path(cfg['blender']),'--background','--factory-startup','--python',ROOT/'scripts/blender_render.py','--',job],timeout=1800)


def criticize(run_id, folder, refs, metrics, prompt, cfg):
    # Include held-out views, not just the image that generated the mesh.
    pairs=metrics['pairs'];indexes=range(len(pairs))
    selected=[]
    for i in indexes:
        p=pairs[i]
        selected.append({'reference_index':p['reference_index'],'reference':refs[p['reference_index']]['prepared'],'render':str(folder/'renders'/p['render']),'camera':{'azimuth_deg':p.get('azimuth_deg'),'elevation_deg':p.get('elevation_deg')}})
    views=json.loads((folder/'renders/views.json').read_text())
    audit=[dict(v,path=str(folder/'renders'/v['file'])) for v in views if v['azimuth_deg'] in (0,90,180,270) and v['elevation_deg']==0]
    audit.extend(dict(v,path=str(folder/'renders'/v['file'])) for v in views if abs(v['elevation_deg'])==90)
    geometry=metrics.get('geometry',{})
    catalog=None
    mesh_path=folder/'mesh.obj'
    report_file=folder/'evaluated_mesh.json'
    if report_file.exists():mesh_path=Path(json.loads(report_file.read_text())['mesh'])
    if mesh_path.is_file() and metrics.get('pairs') and all(p.get('reference_path') for p in metrics['pairs']):
        from gai.shape_evidence import build_catalog
        catalog=build_catalog(mesh_path,folder,metrics['pairs'])
        excluded={tuple(x) for x in cfg.get('_excluded_edits',[])}
        excluded_operations=set(cfg.get('_excluded_operations',[]))
        generation_file=folder.parent/'generation_job.json'
        can_regenerate=generation_file.is_file() and json.loads(generation_file.read_text()).get('backend')=='hunyuan'
        for candidate in catalog['proposals']:
            candidate['permitted_operations']=[op for op in ['fit_silhouette','local_smooth']+(['regenerate'] if can_regenerate else []) if (op,candidate['id']) not in excluded and op not in excluded_operations]
        catalog['proposals']=[p for p in catalog['proposals'] if p['permitted_operations']]
        evidence_views={v['reference_index']:v for v in catalog['views']}
        for pair in selected:
            v=evidence_views.get(pair['reference_index'])
            if v:pair.update(reference=v['target'],difference=v['difference'])
    output=Path(cfg.get('_critic_out',folder));output.mkdir(parents=True,exist_ok=True)
    job=output/'critic_job.json'
    save_json(job,{'model':cfg['qwen_model'],'revision':cfg.get('qwen_revision'),'pairs':selected,'audit_views':audit,'geometry':geometry,'shape_catalog':catalog,'base_description':prompt,'out':str(output),'max_pixels':cfg.get('qwen_max_pixels',384*384),
                   'mode':'choose' if '_reuse_observations' in cfg else 'critic','observations':cfg.get('_reuse_observations',[])})
    command(run_id,[local_path(cfg['qwen_python']),ROOT/'gai/workers/qwen_worker.py',job])
    data=json.loads((output/'critique.json').read_text())
    # The CLIP source is the caller's fixed description, not a fresh description
    # of the desired reference that the VLM may accidentally repeat as both ends.
    data['base_description']=prompt
    data.setdefault('edit_plan',{'operation':'none','reason_ru':'VLM не вернула структурированный план'})
    try:data['edit_plan']=EditPlan.model_validate(data['edit_plan']).model_dump()
    except (ValueError,TypeError):data['edit_plan']=EditPlan(reason_ru='Некорректный план VLM; изменение не выполняется').model_dump()
    if geometry:data=enforce_geometry_warning(data,geometry)
    critique=Critique.model_validate(data)
    save_json(output/'critique.json',dict(critique.model_dump(),peak_vram_mb=data.get('peak_vram_mb')))
    return critique


def verify_edit(run_id,before,after,refs,report,proposal,cfg):
    review=after/'verification';review.mkdir(exist_ok=True)
    job=review/'job.json'
    pairs=[{'reference_index':p['reference_index'],'reference':refs[p['reference_index']]['prepared'],
            'before':str(before/'renders'/p['render']),'after':str(after/'renders'/p['render'])} for p in report['pairs']]
    if proposal.edit_plan.evidence_basis=='silhouette_catalog':
        from PIL import Image
        region_map={r.reference_index:r.bbox for r in proposal.edit_plan.regions}
        for pair in pairs:
            idx=pair['reference_index']
            target=before/'shape_evidence'/f'target_{idx:02d}.png'
            if target.is_file():pair['reference']=str(target)
            box=region_map.get(idx)
            for key in ['reference','before','after']:
                im=Image.open(pair[key]).convert('RGBA')
                if box:
                    x0,y0,x1,y1=box;cx=(x0+x1)/2;cy=(y0+y1)/2;radius=max(x1-x0,y1-y0)*.65
                    bounds=(max(0,cx-radius),max(0,cy-radius),min(1,cx+radius),min(1,cy+radius))
                    im=im.crop(tuple(round(v*(im.width if i%2==0 else im.height)) for i,v in enumerate(bounds)))
                im=im.resize((max(1,im.width*384//max(im.size)),max(1,im.height*384//max(im.size))))
                destination=review/f'view_{idx:02d}_{key}.png';im.save(destination);pair[key]=str(destination)
    evidence=[r.reference_index for r in proposal.edit_plan.regions] or [p['reference_index'] for p in pairs]
    if proposal.edit_plan.operation=='regenerate':evidence=[p['reference_index'] for p in pairs]
    save_json(job,{'mode':'verify','model':cfg.get('qwen_review_model',cfg['qwen_model']),'revision':cfg.get('qwen_review_revision',cfg.get('qwen_revision')),
                   'out':str(review),'max_pixels':cfg.get('qwen_review_max_pixels',cfg.get('qwen_max_pixels',384*384)),'pairs':pairs,'proposal':proposal.model_dump(),'evidence_indexes':evidence})
    command(run_id,[local_path(cfg['qwen_python']),ROOT/'gai/workers/qwen_worker.py',job])
    # Revalidate the raw per-view assessments, not just an untrusted accepted flag.
    from gai.review import summarize_review
    data=json.loads((review/'verification.json').read_text())
    result=summarize_review(data['views'],evidence)
    result.update({k:data[k] for k in ['model','revision','peak_vram_mb','protocol'] if k in data})
    if {v['reference_index'] for v in result['views']} != {p['reference_index'] for p in pairs}:
        raise ValueError('Независимая проверка не охватила все фото')
    save_json(review/'verification.json',result)
    return result


def edit(run_id,mesh,folder,critique,cfg):
    if critique.edit_plan is not None:
        source=Path(cfg.get('_edit_source',folder.parent/f'iteration_{int(folder.name.split("_")[-1])-1:02d}'))
        if critique.edit_plan.operation=='fit_silhouette':
            from gai.shape_fit import fit_mesh
            candidate=fit_mesh(mesh,source,folder/'local_edit',critique.edit_plan.proposal_id)
        elif critique.edit_plan.operation=='local_smooth' and critique.edit_plan.evidence_basis=='silhouette_catalog':
            from gai.shape_fit import smooth_mesh
            candidate=smooth_mesh(mesh,source,folder/'local_edit',critique.edit_plan.proposal_id)
        elif critique.edit_plan.operation=='regenerate':
            # A new candidate from the same inputs; never pretend a deformation adds topology.
            job_data=json.loads((folder.parent/'generation_job.json').read_text())
            job_data.update(out=str(folder/'regeneration'),seed=int(job_data['seed'])+int(folder.name.split('_')[-1]))
            from gai.regeneration import plan_regeneration
            ref_data=json.loads((source/'references.json').read_text()) if (source/'references.json').is_file() else json.loads((source.parent/'iteration_00/references.json').read_text())
            regeneration=plan_regeneration(job_data,ref_data,json.loads((source/'metrics.json').read_text()))
            if job_data.get('backend')=='hunyuan':
                job_data.update(selected=regeneration['selected'],resolution=max(job_data.get('resolution',256),512))
            if job_data.get('backend','sf3d')=='sf3d':job_data['retry']=False
            job=folder/'regeneration_job.json';save_json(job,job_data)
            backend=job_data.get('backend','sf3d')
            if backend=='da3':raise UnsupportedEdit('DA3 не поддерживает содержательную повторную генерацию по другому seed')
            worker='multiview_worker.py' if backend in {'hunyuan','da3'} else 'sf3d_worker.py'
            command(run_id,[local_path(cfg[backend+'_python']),ROOT/'gai/workers'/worker,job],timeout=7200)
            candidate=folder/'regeneration/mesh.obj'
            canonical_mesh(folder/'regeneration/generated.glb',candidate,cfg['max_faces'])
            chosen=job_data.get('selected',{'front':0})
            generation_views=list(chosen.values()) if isinstance(chosen,dict) else list(chosen)
            import math
            restored=trimesh.load(candidate,force='mesh',process=False)
            if regeneration['yaw_degrees']:
                restored.apply_transform(trimesh.transformations.rotation_matrix(math.radians(regeneration['yaw_degrees']),[0,1,0]))
            from gai.regeneration import register_translation
            regeneration['translation']=register_translation(restored,source,generation_views)
            restored.export(candidate)
            save_json(folder/'edit_report.json',{'backend':'regeneration','plan':critique.edit_plan.model_dump(),'seed':job_data['seed'],
                'generation_views':generation_views,'resolution':job_data.get('resolution'),'registration':regeneration})
        else:
            candidate=apply_plan(mesh,critique.edit_plan,source,folder/'local_edit')
        validate_candidate(candidate)
        return candidate
    job=folder/'edit_job.json'
    save_json(job,{'repo':str(local_path(cfg['textdeformer_repo'])),'mesh':str(mesh),'out':str(folder/'textdeformer'),'base_description':critique.base_description,'target_description':critique.target_description,'epochs':cfg['textdeformer_epochs'],'batch_size':cfg['textdeformer_batch_size'],'train_res':cfg['textdeformer_train_res']})
    command(run_id,[local_path(cfg['textdeformer_python']),ROOT/'gai/workers/edit_worker.py',job],cwd=local_path(cfg['textdeformer_repo']),timeout=7200)
    candidate=folder/'textdeformer/mesh_final/mesh.obj'
    validate_candidate(candidate)
    return candidate


def execute(run_id, resume_critique=False):
    folder=run_dir(run_id)
    cfg=read_run(run_id).get('config') or config()
    lock=(RUNS/'.gpu.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        update_run(run_id,status='failed',error='GPU занята другим процессом проекта');lock.close();return
    try:
        state=read_run(run_id);params=state['parameters'];refs=state['references']
        if resume_critique:
            if (state['status'] not in {'failed','completed'} and not (state['status']=='queued' and state.get('initial_mesh_import'))) or (state['status']=='failed' and state.get('stage')!='critique') or state.get('iteration')!=0 or len(state.get('iterations',[]))>1:
                raise ValueError('Повтор оценки поддерживается только для сохранённой исходной модели')
            initial=folder/'iteration_00'
            mesh=ROOT/state['preview_mesh']
            refs=json.loads((initial/'references.json').read_text())
            meta=json.loads((initial/'mesh_info.json').read_text())
            backend=meta.get('backend','Stable Fast 3D')
            if not mesh.is_file() or not (initial/'renders/views.json').is_file():
                raise ValueError('Нет сохранённой сетки или рендеров для повторной оценки')
            update_run(run_id,status='running',stage='critique',error=None)
            event(run_id,'Повторная оценка сохранённой модели: генерация и рендеры не повторяются')
        else:
            update_run(run_id,status='running',stage='generate')
            initial=folder/'iteration_00';initial.mkdir(exist_ok=True)
            generator=params.get('generator','auto')
            if generator in {'hunyuan','da3'}:
                from gai.multiview import select_views
                selected=select_views(refs,generator,params.get('view_sides'))
                backend={'hunyuan':'Hunyuan3D-2mv','da3':'DA3 Base + Poisson'}[generator]
                event(run_id,backend+': генерация по реальным ракурсам; отдельные фото оставлены для проверки')
                job=folder/'generation_job.json'
                payload={'backend':generator,'repo':str(local_path(cfg[generator+'_repo'])),
                    'model':cfg[generator+'_model'],'revision':cfg.get(generator+'_revision'),'out':str(initial),'references':refs,
                    'background':params['background'],'seed':params['seed'],'selected':selected,
                    'resolution':cfg[generator+'_resolution']}
                if generator=='hunyuan':payload.update(subfolder=cfg['hunyuan_subfolder'],steps=cfg['hunyuan_steps'])
                save_json(job,payload)
                command(run_id,[local_path(cfg[generator+'_python']),ROOT/'gai/workers/multiview_worker.py',job],timeout=7200)
                refs=json.loads((initial/'references.json').read_text())
                meta=canonical_mesh(initial/'generated.glb',initial/'mesh.obj',cfg['max_faces'])
                mesh=initial/'mesh.obj'
                save_json(initial/'mesh_info.json',dict(meta,backend=backend))
                if not meta['geometry']['watertight'] or not meta['can_deform']:
                    event(run_id,'Проверка структуры: сетка незамкнута или содержит несколько частей. Результат требует проверки; артефакты сохранены.')
            else:
                use_multiview=params.get('generator','auto')!='sf3d' and len(refs)>=5 and all(r.get('azimuth_deg') is not None for r in refs)
                event(run_id,'SF3D: подготовка масок и генерация по первому выбранному изображению')
                job=folder/'generation_job.json'
                save_json(job,{'repo':str(local_path(cfg['sf3d_repo'])),'model':cfg['sf3d_model'],'revision':cfg.get('sf3d_revision'),'seed':params['seed'],'out':str(initial),'references':refs,'background':params['background'],'retry':not use_multiview})
                command(run_id,[local_path(cfg['sf3d_python']),ROOT/'gai/workers/sf3d_worker.py',job])
                refs=json.loads((initial/'references.json').read_text())
                quality_file=initial/'generation_quality.json'
                if quality_file.exists():
                    quality=json.loads(quality_file.read_text())
                    if len(quality['candidates'])>1:
                        event(run_id,f"SF3D: проверено {len(quality['candidates'])} отдельных ракурсов; выбран №{quality['selected_reference_index']} по эвристике толщины. Это не оценка точности 3D.")
                meta=canonical_mesh(initial/'generated.glb',initial/'mesh.obj',cfg['max_faces']);save_json(initial/'mesh_info.json',meta)
                mesh=initial/'mesh.obj'
                save_json(initial/'sf3d_mesh_info.json',meta)
                backend='Stable Fast 3D'
                if use_multiview:
                    from gai.reconstruction import visual_hull
                    event(run_id,'Многовидовая реконструкция: известные азимуты, часть фото оставлена для проверки')
                    reconstruction=initial/'reconstruction';reconstruction.mkdir(exist_ok=True)
                    original_generation_flags=[ref.get('used_for_generation',i==0) for i,ref in enumerate(refs)]
                    # Do not reuse the reserved photos in SF3D retries or hull construction.
                    heldout=max(1,len(refs)//4)
                    heldout_indexes=set(np.linspace(1,len(refs)-1,heldout,dtype=int).tolist())
                    training=[ref for i,ref in enumerate(refs) if i not in heldout_indexes]
                    for i,ref in enumerate(refs):ref['used_for_generation']=i not in heldout_indexes
                    try:
                        reconstructed=visual_hull(training)
                        reconstructed.export(reconstruction/'visual_hull.glb')
                        shutil.copy2(mesh,initial/'sf3d.obj')
                        meta=canonical_mesh(reconstruction/'visual_hull.glb',reconstruction/'mesh.obj',cfg['max_faces'])
                        mesh=reconstruction/'mesh.obj';backend='Многовидовая реконструкция по силуэтам'
                        save_json(reconstruction/'reconstruction.json',{'method':'orthographic visual hull','training_references':[r['prepared'] for r in training],
                            'heldout_references':[r['prepared'] for i,r in enumerate(refs) if i in heldout_indexes],'elevation_deg':0,'camera_estimated':True,
                            'mesh_info':meta,'limitations':'Известны относительные азимуты; масштаб по высоте и камера приблизительные. Впадины и скрытые детали не восстанавливаются.'})
                        save_json(initial/'references.json',refs)
                        event(run_id,'Восстановлен объём по нескольким реальным силуэтам; исходная SF3D-сетка сохранена отдельно')
                        source_preview=initial/'sf3d_evaluation';source_preview.mkdir(exist_ok=True)
                        render(run_id,initial/'sf3d.obj',source_preview,dict(cfg,render_elevations=[0],render_step_deg=90))
                        source_report=align(refs,source_preview/'renders')
                        for pair in source_report['pairs']:
                            pair['reference_path']=str(Path(refs[pair['reference_index']]['prepared']).relative_to(ROOT))
                            pair['used_for_generation']=pair['reference_index']==0
                        source_report.update(validation_summary(source_report))
                        source_report['geometry']=geometry_report(trimesh.load(initial/'sf3d.obj',force='mesh'))
                        save_json(source_preview/'comparison.json',{'path':str(source_preview.relative_to(ROOT)),
                            'mesh':str((initial/'sf3d.obj').relative_to(ROOT)),'metrics':source_report,
                            'note':'Четыре диагностических вида; для сопоставления используется ближайший силуэт, камера приблизительная.'})
                    except ValueError as exc:
                        if params.get('generator')=='multiview':raise
                        mesh=initial/'mesh.obj';meta=json.loads((initial/'sf3d_mesh_info.json').read_text());backend='Stable Fast 3D'
                        for i,ref in enumerate(refs):ref['used_for_generation']=original_generation_flags[i]
                        save_json(initial/'references.json',refs)
                        event(run_id,'Многовидовая реконструкция недоступна: '+str(exc)+'. Сохранён результат SF3D.')
                save_json(initial/'mesh_info.json',dict(meta,backend=backend))
        baseline_pairs=None;iterations=[];current_description=params['prompt'];excluded_edits=[];excluded_operations=[]
        update_run(run_id,preview_mesh=str(mesh.relative_to(ROOT)),source_glb=str((initial/'generated.glb').relative_to(ROOT)))
        for number in range(params['iterations']+1):
            current=folder/f'iteration_{number:02d}';current.mkdir(exist_ok=True)
            update_run(run_id,stage='render',iteration=number)
            event(run_id,f'Итерация {number}: используются сохранённые рендеры' if resume_critique and number==0 else f'Итерация {number}: EEVEE, фиксированная камера и clay-материал')
            if not (resume_critique and number==0):
                render_cfg=cfg
                if number and all('azimuth_deg' in p and 'elevation_deg' in p for p in previous_report['pairs']):
                    render_cfg=dict(cfg,_render_views=previous_report['pairs']+previous_report.get('audit_views',[]))
                render(run_id,mesh,current,render_cfg)
            generation_report=current/'edit_report.json'
            if generation_report.is_file():
                inputs=json.loads(generation_report.read_text()).get('generation_views',[])
                for i,ref in enumerate(refs):ref['used_for_generation']=ref.get('used_for_generation',i==0) or i in inputs
            report=align(refs,current/'renders',baseline_pairs)
            for pair in report['pairs']:
                pair['reference_azimuth_deg']=refs[pair['reference_index']].get('azimuth_deg')
                pair['used_for_generation']=refs[pair['reference_index']].get('used_for_generation',pair['reference_index']==0)
                pair['reference_path']=str(Path(refs[pair['reference_index']]['prepared']).relative_to(ROOT)) if Path(refs[pair['reference_index']]['prepared']).is_absolute() else refs[pair['reference_index']]['prepared']
            views=json.loads((current/'renders/views.json').read_text()) if (current/'renders/views.json').exists() else []
            report['audit_views']=[dict(v,path=str((current/'renders'/v['file']).relative_to(ROOT))) for v in views if (v['azimuth_deg'] in (0,90,180,270) and v['elevation_deg']==0) or abs(v['elevation_deg'])==90]
            report['geometry']=geometry_report(trimesh.load(mesh,force='mesh',process=True))
            if number and previous_critique.edit_plan and previous_critique.edit_plan.operation=='fit_silhouette':
                trained={r.reference_index for r in previous_critique.edit_plan.regions}
                for pair in report['pairs']:pair['used_for_edit']=pair['reference_index'] in trained
            prior_inputs={p['reference_index'] for p in previous_report['pairs'] if p.get('used_for_edit',False)} if number else {i for i,r in enumerate(refs) if r.get('used_for_edit',False)}
            for pair in report['pairs']:
                pair['used_for_edit']=pair.get('used_for_edit',False) or pair['reference_index'] in prior_inputs
            report.update(validation_summary(report))
            save_json(current/'metrics.json',report)
            save_json(current/'evaluated_mesh.json',{'mesh':str(mesh)})
            if baseline_pairs is None:baseline_pairs=report['pairs']
            update_run(run_id,stage='critique')
            critique=criticize(run_id,current,refs,report,current_description,cfg)
            accepted=True;reason='Исходная модель'
            verification=None
            if number:
                localized=previous_critique.edit_plan is not None
                control_before=previous_report
                if previous_critique.edit_plan and previous_critique.edit_plan.operation=='fit_silhouette':
                    trained={r.reference_index for r in previous_critique.edit_plan.regions}
                    control_before=dict(previous_report,pairs=[dict(p,used_for_edit=p.get('used_for_edit',False) or p['reference_index'] in trained) for p in previous_report['pairs']])
                    report['edit_training_views']=sorted(trained)
                if previous_critique.edit_plan and previous_critique.edit_plan.operation=='regenerate' and generation_report.is_file():
                    inputs=set(json.loads(generation_report.read_text()).get('generation_views',[]))
                    control_before=dict(previous_report,pairs=[dict(p,used_for_generation=p.get('used_for_generation',p['reference_index']==0) or p['reference_index'] in inputs) for p in previous_report['pairs']])
                accepted,reason=acceptance(control_before,report,cfg['acceptance_margin'],require_improvement=(not localized or previous_critique.edit_plan.operation=='regenerate'))
                fitting=current/'local_edit/edit_report.json'
                if fitting.is_file():
                    fit_data=json.loads(fitting.read_text())
                    if 'after_fixed_frame_iou' in fit_data:
                        report['fixed_frame_fit']=fit_data
                        control=[p['reference_index'] for p in control_before['pairs'] if not p.get('used_for_generation',p['reference_index']==0) and not p.get('used_for_edit',False)]
                        changes=[fit_data['after_fixed_frame_iou'][str(i)]-fit_data['before_fixed_frame_iou'][str(i)] for i in control]
                        if not changes or np.mean(changes)<-.005 or min(changes)<-.03:
                            accepted=False;reason+='; ухудшились независимые контуры в фиксированной системе камеры'
                if localized:
                    update_run(run_id,stage='verify')
                    event(run_id,'Независимая VLM: сравнение референсов с моделью до и после правки')
                    verification=verify_edit(run_id,previous_folder,current,refs,report,previous_critique,cfg)
                    accepted=accepted and verification['accepted']
                    reason+='; '+verification['reason_ru']
                if critique.camera_mismatch:accepted=False;reason+='; критик обнаружил несогласование камеры'
            save_json(current/'metrics.json',report)
            item={'backend':backend,'number':number,'path':str(current.relative_to(ROOT)),'mesh':str(mesh.relative_to(ROOT)),'metrics':report,'critique':critique.model_dump(),'accepted':accepted,'decision':reason}
            if number:item['edit_source_mesh']=str(best_mesh.relative_to(ROOT))
            if verification is not None:item['verification']=verification
            for edit_report in (current/'local_edit/edit_report.json',current/'edit_report.json'):
                if edit_report.exists():item['edit_report']=json.loads(edit_report.read_text())
            iterations.append(item);update_run(run_id,iterations=iterations)
            event(run_id,reason)
            if accepted:
                best_mesh=mesh;best_report=report;best_folder=current;best_critique=critique;excluded_edits=[];excluded_operations=[]
                # Standalone geometry export, independent of TextDeformer sidecar files.
                trimesh.load(best_mesh,force='mesh',process=False).export(folder/'best.glb')
                update_run(run_id,best_mesh=str((folder/'best.glb').relative_to(ROOT)))
            if not accepted:
                event(run_id,'Правка отклонена; сохранён предыдущий результат')
                rejected=previous_critique.edit_plan
                if rejected and rejected.evidence_basis=='silhouette_catalog' and number<params['iterations']:
                    excluded_edits.append((rejected.operation,rejected.proposal_id));excluded_operations.append(rejected.operation)
                    next_folder=folder/f'iteration_{number+1:02d}';next_folder.mkdir(exist_ok=True)
                    event(run_id,'VLM выбирает другую операцию или область с учётом отклонённого кандидата')
                    alternative=criticize(run_id,best_folder,refs,best_report,params['prompt'],
                        dict(cfg,_critic_out=next_folder/'planning',_reuse_observations=best_critique.view_assessments,_excluded_edits=excluded_edits,_excluded_operations=excluded_operations))
                    if alternative.should_edit and alternative.edit_plan and alternative.edit_plan.operation!='none' and alternative.confidence>=.65:
                        previous_report=best_report;previous_folder=best_folder;previous_critique=alternative
                        update_run(run_id,stage='edit')
                        try:mesh=edit(run_id,best_mesh,next_folder,alternative,dict(cfg,_edit_source=best_folder))
                        except UnsupportedEdit as exc:
                            event(run_id,'Альтернативная правка недоступна: '+str(exc));break
                        current_description=params['prompt']
                        continue
                break
            stop_reason=None
            if number>=params['iterations']:
                stop_reason='Достигнут выбранный лимит итераций'
            elif critique.should_edit and critique.edit_type in {'remove','repair'} and critique.edit_plan is None:
                stop_reason='VLM запросила удаление лишних поверхностей или ремонт сетки. TextDeformer не меняет топологию; операция не подменяется добавлением объёма.'
            elif critique.camera_mismatch:
                stop_reason='Правка пропущена: критик указал несогласование камеры'
            elif not critique.should_edit:
                stop_reason='Критик не запросил изменение геометрии'
            elif critique.confidence<0.65:
                stop_reason='Правка пропущена: низкая уверенность критика'
            if stop_reason:
                event(run_id,stop_reason);break
            if critique.edit_plan is not None and critique.edit_plan.operation in {'local_scale','local_smooth','remove_region'} and not critique.view_assessments:
                event(run_id,'Правка пропущена: нет наблюдений, подтверждающих локальную область');break
            if critique.edit_plan is not None and critique.edit_plan.operation=='none':
                event(run_id,'Правка пропущена: '+critique.edit_plan.reason_ru);break
            if critique.edit_plan is None and not meta['can_deform']:
                event(run_id,'Редактирование пропущено: несколько крупных несвязных частей. Требуется подготовка связной сетки.');break
            if critique.edit_plan is None and re.findall(r'\w+',critique.target_description.casefold())==re.findall(r'\w+',critique.base_description.casefold()):
                event(run_id,'Правка пропущена: одинаковые исходное и целевое описания');break
            update_run(run_id,stage='edit')
            event(run_id,('Локальный план: '+critique.edit_plan.operation) if critique.edit_plan else 'TextDeformer: '+critique.target_description)
            next_folder=folder/f'iteration_{number+1:02d}';next_folder.mkdir(exist_ok=True)
            previous_report=report;previous_critique=critique;previous_folder=current
            try:mesh=edit(run_id,mesh,next_folder,critique,cfg)
            except UnsupportedEdit as exc:
                save_json(next_folder/'edit_unavailable.json',{'reason_ru':str(exc),'plan':critique.edit_plan.model_dump() if critique.edit_plan else None})
                event(run_id,'Локальная правка не выполнена: '+str(exc));break
            current_description=params['prompt'] if critique.edit_plan else critique.target_description
        update_run(run_id,status='completed',stage='done')
        event(run_id,'Запуск завершён. Исходные изображения и все промежуточные результаты сохранены.')
    except Cancelled:
        update_run(run_id,status='cancelled',stage='stopped');event(run_id,'Запуск остановлен')
    except Exception as exc:
        (folder/'error.txt').write_text(traceback.format_exc())
        update_run(run_id,status='failed',error=str(exc));event(run_id,str(exc))
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
