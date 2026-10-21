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


def test_camera_excuse_cannot_hide_measured_flatness():
    report=geometry_report(trimesh.creation.box(extents=[1,2,.03]))
    critique={'summary_ru':'Всё согласовано','camera_mismatch':True,'issues':[],
              'should_edit':True,'base_description':'a cat','target_description':'a fat cat'}
    result=enforce_geometry_warning(critique,report)
    assert 'малая толщина' in result['summary_ru']
    assert result['issues'] and not result['should_edit']
    assert result['target_description']=='a cat'
