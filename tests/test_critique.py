from gai.pipeline import Critique


def test_no_edit_never_leaves_an_unrequested_target_description():
    c=Critique(summary_ru='No issues',camera_mismatch=False,confidence=.9,should_edit=False,issues=[],base_description='a duck',target_description='a giraffe')
    assert c.target_description=='a duck'


def test_camera_mismatch_blocks_a_semantic_edit():
    c=Critique(summary_ru='Camera mismatch',camera_mismatch=True,confidence=.9,should_edit=True,issues=[{'part':'head'}],base_description='a duck',target_description='a duck with a larger head')
    assert not c.should_edit and c.target_description==c.base_description


def test_removal_is_preserved_as_vlm_requested():
    c=Critique(summary_ru='Лишняя плоскость',camera_mismatch=False,confidence=.8,should_edit=True,
               edit_type='remove',issues=[{'part':'extra sheet'}],base_description='a frog',target_description='a frog without an extra sheet')
    assert c.should_edit and c.edit_type=='remove'


def test_explicit_none_cannot_trigger_deformation():
    c=Critique(summary_ru='No supported change',camera_mismatch=False,confidence=.8,should_edit=True,
               edit_type='none',issues=[{'part':'uncertain'}],base_description='a frog',target_description='a fat frog')
    assert not c.should_edit and c.target_description=='a frog'
