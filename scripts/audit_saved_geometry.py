"""Add geometry diagnostics to completed local runs; preserve previous reports."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trimesh
from gai.core import ROOT,RUNS,save_json,now
from gai.geometry import geometry_report,enforce_geometry_warning


def main():
    for path in RUNS.glob('*/state.json'):
        state=json.loads(path.read_text())
        if state['status']!='completed' or not state.get('iterations'):continue
        changed=False
        for item in state['iterations']:
            if 'geometry' in item['metrics']:continue
            mesh=ROOT/item['mesh']
            if not mesh.exists():continue
            if not changed:
                shutil.copy2(path,path.with_name('state.before_volume_audit.json'))
            changed=True
            report=geometry_report(trimesh.load(mesh,force='mesh',process=False))
            item['metrics']['geometry']=report
            folder=ROOT/item['path']
            views_path=folder/'renders/views.json'
            if views_path.exists():
                views=json.loads(views_path.read_text())
                item['metrics']['audit_views']=[dict(v,path=str((folder/'renders'/v['file']).relative_to(ROOT))) for v in views if v['azimuth_deg'] in (0,90,180,270) and v['elevation_deg']==0]
            for filename in ['metrics.json','critique.json']:
                source=folder/filename
                backup=source.with_name(filename+'.before_volume_audit')
                if source.exists() and not backup.exists():shutil.copy2(source,backup)
            item['critique']=enforce_geometry_warning(item['critique'],report)
            save_json(folder/'metrics.json',item['metrics'])
            save_json(folder/'critique.json',item['critique'])
        if changed:
            state['events'].append({'time':now(),'message':'Добавлена независимая проверка толщины сохранённой геометрии. Это не повторная генерация и не новая оценка VLM; предыдущие отчёты сохранены.'})
            state['updated_at']=now();save_json(path,state)
            print(state['id'],'geometry diagnostics added')


if __name__=='__main__':main()
