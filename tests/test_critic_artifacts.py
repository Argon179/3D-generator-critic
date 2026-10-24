import json
import trimesh
from PIL import Image
from gai.core import save_json


def test_new_critic_editor_and_verifier_artifacts_are_separate(tmp_path,monkeypatch):
    import gai.artifacts as artifacts
    ident='aaaa11223344';folder=tmp_path/'runs'/ident
    save_json(folder/'state.json',{'id':ident,'parameters':{},'references':[],'events':[],'iterations':[]})
    save_json(folder/'iteration_00/view_00.json',{'observation_ru':'test'})
    save_json(folder/'iteration_00/view_assessments.json',[])
    save_json(folder/'iteration_01/local_edit/edit_report.json',{'backend':'local'})
    trimesh.creation.box().export(folder/'iteration_01/local_edit/mesh.obj')
    Image.new('RGB',(8,8)).save(folder/'iteration_01/local_edit/region_00.png')
    save_json(folder/'iteration_01/verification/verification.json',{'accepted':False})
    monkeypatch.setattr(artifacts,'ROOT',tmp_path)
    monkeypatch.setattr(artifacts,'run_dir',lambda _:folder)
    monkeypatch.setattr(artifacts,'read_run',lambda _:json.loads((folder/'state.json').read_text()))
    stages={s['id']:s for s in artifacts.stage_inventory(ident)['stages']}
    assert len(stages['critic_0']['files'])==2
    assert stages['edit_1']['models'] and stages['edit_1']['images']
    assert stages['verify_1']['details']['iteration_01/verification/verification.json']['accepted'] is False
