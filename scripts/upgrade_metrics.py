"""Recompute diagnostics for saved runs without changing their meshes or VLM verdicts."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trimesh
from gai.core import ROOT,RUNS,save_json,now
from gai.geometry import geometry_report
from gai.metrics import align,validation_summary


def main():
    for path in RUNS.glob('*/state.json'):
        state=json.loads(path.read_text())
        if state['status']!='completed':continue
        refs_file=path.parent/'iteration_00/references.json'
        if not refs_file.exists():continue
        refs=json.loads(refs_file.read_text());changed=False
        quality_file=path.parent/'iteration_00/generation_quality.json'
        tried={c['reference_index'] for c in json.loads(quality_file.read_text())['candidates']} if quality_file.exists() else {0}
        for item in state.get('iterations',[]):
            if 'heldout' in item['metrics'] and 'flat_cap_fraction' in item['metrics'].get('geometry',{}):continue
            folder=ROOT/item['path']
            if not (folder/'renders/views.json').exists():continue
            if not changed:shutil.copy2(path,path.with_name('state.before_metrics_v2.json'))
            changed=True
            report=align(refs,folder/'renders',item['metrics']['pairs'])
            for pair in report['pairs']:
                ref=refs[pair['reference_index']]
                original=int(Path(ref['prepared']).stem.rsplit('_',1)[-1])
                pair['used_for_generation']=ref.get('used_for_generation',original in tried)
                pair['reference_path']=str(Path(ref['prepared']).relative_to(ROOT))
            report['geometry']=geometry_report(trimesh.load(ROOT/item['mesh'],force='mesh',process=False))
            report['audit_views']=item['metrics'].get('audit_views',[])
            report.update(validation_summary(report));item['metrics']=report
            original_file=folder/'metrics.json'
            if original_file.exists():shutil.copy2(original_file,folder/'metrics.before_v2.json')
            save_json(original_file,report)
        if changed:
            state['events'].append({'time':now(),'message':'Дополнены метрики контура и 3D-структуры. Модели и исходные вердикты VLM сохранены.'})
            state['updated_at']=now();save_json(path,state);print(state['id'],'upgraded')


if __name__=='__main__':main()
