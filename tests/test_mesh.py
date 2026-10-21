import numpy as np
import trimesh
from gai.pipeline import canonical_mesh, validate_candidate


def test_glb_scene_transform_is_baked_before_canonicalization(tmp_path):
    box=trimesh.creation.box(extents=[1,2,3])
    matrix=np.eye(4);matrix[:3,3]=[10,20,30]
    scene=trimesh.Scene();scene.add_geometry(box,transform=matrix)
    source=tmp_path/'source.glb';scene.export(source)
    dest=tmp_path/'mesh.obj';info=canonical_mesh(source,dest,1000)
    output=trimesh.load(dest,force='mesh')
    assert np.allclose(output.bounds.mean(axis=0),0,atol=1e-6)
    assert np.isclose(output.extents.max(),2)
    assert np.allclose(info['source_center'],[10,20,30])
    validate_candidate(dest)
