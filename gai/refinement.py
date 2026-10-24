"""Create a new, traceable refinement run from a saved accepted mesh."""
import json,shutil,uuid
from pathlib import Path
from gai.core import ROOT,RUNS,config,now,read_run,run_dir,save_json


def create_refinement(parent_id,iterations=3):
    parent=read_run(parent_id)
    if parent['status'] not in {'completed','failed','cancelled'}:raise ValueError('Сначала дождитесь завершения исходного запуска')
    accepted=[i for i in parent.get('iterations',[]) if i.get('accepted')]
    if not accepted:raise ValueError('Нет сохранённой исходной или принятой модели')
    best=accepted[-1];source=ROOT/best['mesh'];render_folder=ROOT/best['path']/'renders'
    refs_file=run_dir(parent_id)/'iteration_00/references.json'
    if not source.is_file() or not (render_folder/'views.json').is_file() or not refs_file.is_file():
        raise ValueError('Нет сетки, подготовленных фото или камер для уточнения')
    ident=uuid.uuid4().hex[:12];folder=RUNS/ident;initial=folder/'iteration_00';initial.mkdir(parents=True)
    try:
        refs=json.loads(refs_file.read_text());raw=[];prepared=[]
        generation_flags={p['reference_index']:p.get('used_for_generation',p['reference_index']==0) for p in best.get('metrics',{}).get('pairs',[])}
        edit_flags={p['reference_index']:p.get('used_for_edit',False) for p in best.get('metrics',{}).get('pairs',[])}
        for i,(original,ref) in enumerate(zip(parent['references'],refs)):
            path=folder/'source_images'/f'view_{i:02d}.png';path.parent.mkdir(exist_ok=True)
            shutil.copy2(original['path'],path)
            dest=initial/f'reference_{i:02d}.png';shutil.copy2(ref['prepared'],dest)
            raw.append(dict(original,path=str(path)))
            prepared.append(dict(ref,path=str(path),prepared=str(dest),used_for_generation=generation_flags.get(i,ref.get('used_for_generation',i==0)),used_for_edit=edit_flags.get(i,ref.get('used_for_edit',False))))
        shutil.copy2(source,initial/'mesh.obj');shutil.copytree(render_folder,initial/'renders')
        # Keep canonical coordinates and topology; do not re-normalize the accepted mesh.
        import trimesh
        from gai.geometry import geometry_report
        mesh=trimesh.load(source,force='mesh',process=True)
        mesh.export(initial/'generated.glb');shutil.copy2(initial/'generated.glb',folder/'best.glb')
        save_json(initial/'references.json',prepared)
        save_json(initial/'mesh_info.json',{'backend':best.get('backend','Imported mesh'),'can_deform':mesh.body_count==1,
            'geometry':geometry_report(mesh),'parent_run':parent_id,'parent_mesh':best['mesh'],'coordinates_preserved':True})
        generation=run_dir(parent_id)/'generation_job.json'
        if generation.is_file():
            job=json.loads(generation.read_text());job.update(references=raw,out=str(initial));save_json(folder/'generation_job.json',job)
        params=dict(parent['parameters'],iterations=iterations)
        state={'id':ident,'name':parent['name'].split(' · уточнение')[0]+' · уточнение','created_at':now(),'updated_at':now(),
            'status':'queued','stage':'critique','iteration':0,'initial_mesh_import':True,'parent_run':parent_id,
            'parameters':params,'config':config(),'references':raw,'iterations':[],
            'preview_mesh':str((initial/'mesh.obj').relative_to(ROOT)),'source_glb':str((initial/'generated.glb').relative_to(ROOT)),
            'best_mesh':str((folder/'best.glb').relative_to(ROOT)),
            'events':[{'time':now(),'message':'Уточнение сохранённой модели из запуска '+parent_id+'; генерация не повторяется'}]}
        save_json(folder/'state.json',state)
        return state
    except Exception:
        shutil.rmtree(folder)
        raise
