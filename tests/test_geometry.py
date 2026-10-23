import numpy as np
import trimesh
from gai.geometry import geometry_report, enforce_geometry_warning


def test_flatness_survives_rotation_scale_and_subdivision():
    sheet=trimesh.creation.box(extents=[2,1,.03])
    initial=geometry_report(sheet)
    assert initial['severity']=='severe'
    sheet=sheet.subdivide()
    sheet.apply_transform(trimesh.transformations.rotation_matrix(.73,[1,2,3]))
    sheet.apply_scale(7)
    sheet.apply_translation([100,-5,8])
    assert np.isclose(geometry_report(sheet)['thickness_ratio'],initial['thickness_ratio'])
    assert geometry_report(trimesh.creation.icosphere())['severity']=='none'


def test_metrics_never_override_vlm_edit_decision():
    for mesh in [trimesh.creation.icosphere(),trimesh.creation.box(extents=[1,2,.03])]:
        for operation in ['add','remove','reshape','none']:
            c={'summary_ru':'Лишняя плоская поверхность','issues':[], 'edit_type':operation,
               'should_edit':operation!='none','base_description':'a frog','target_description':'a frog without an extra sheet'}
            assert enforce_geometry_warning(c,geometry_report(mesh))==c
