import numpy as np
import trimesh
from gai.pipeline import canonical_mesh


def test_tiny_floating_artifact_is_removed_before_centering(tmp_path):
    main=trimesh.creation.icosphere(subdivisions=3,radius=1)
    tiny=trimesh.creation.icosphere(subdivisions=0,radius=.01);tiny.apply_translation([10,0,0])
    mesh=trimesh.util.concatenate([main,tiny]);source=tmp_path/'parts.glb';mesh.export(source)
    meta=canonical_mesh(source,tmp_path/'mesh.obj',20000)
    assert meta['can_deform']
    assert meta['removed_fragment_faces']==20
    assert np.allclose(meta['source_center'],[0,0,0])


def test_major_disconnected_parts_are_preserved_but_not_editable(tmp_path):
    first=trimesh.creation.box();second=trimesh.creation.box();second.apply_translation([3,0,0])
    source=tmp_path/'parts.glb';trimesh.util.concatenate([first,second]).export(source)
    meta=canonical_mesh(source,tmp_path/'mesh.obj',1000)
    assert not meta['can_deform']
    assert meta['removed_fragment_faces']==0
    assert meta['faces']==24
