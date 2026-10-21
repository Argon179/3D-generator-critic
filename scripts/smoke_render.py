"""Real EEVEE test, no neural generation or fake critic. Outputs are labelled."""
import sys
import subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trimesh
from gai.core import ROOT, config, local_path, save_json, child_env
from gai.metrics import align

def main():
    cfg=config();out=ROOT/'runs/_smoke';out.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=0.85)
    mesh.vertices[:,1]*=1.15;mesh.export(out/'fixture.obj')
    job=out/'render_job.json'
    save_json(job,{'mesh':str(out/'fixture.obj'),'out':str(out/'renders'),'resolution':192,'step':90,'elevations':[20]})
    subprocess.run([str(local_path(cfg['blender'])),'--background','--factory-startup','--python',str(ROOT/'scripts/blender_render.py'),'--',str(job)],check=True,env=child_env())
    refs=[{'prepared':str(p),'azimuth_deg':int(p.stem.split('_')[1])} for p in sorted((out/'renders').glob('*.png'))]
    metrics=align(refs,out/'renders')
    assert abs(metrics['silhouette_iou']-1)<1e-8
    save_json(out/'test_report.json',{'test':'real EEVEE rendering + silhouette roundtrip; neural models not invoked','metrics':metrics})
    print('PASS: 4 real EEVEE renders; silhouette roundtrip IoU = 1.0')

if __name__=='__main__':main()
