from gai.pipeline import Critique


def test_no_edit_never_leaves_an_unrequested_target_description():
    c=Critique(summary_ru='No issues',camera_mismatch=False,confidence=.9,should_edit=False,issues=[],base_description='a duck',target_description='a giraffe')
    assert c.target_description=='a duck'


def test_camera_mismatch_blocks_a_semantic_edit():
    c=Critique(summary_ru='Camera mismatch',camera_mismatch=True,confidence=.9,should_edit=True,issues=[{'part':'head'}],base_description='a duck',target_description='a duck with a larger head')
    assert not c.should_edit and c.target_description==c.base_description
