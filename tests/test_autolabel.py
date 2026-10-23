import pytest
from gai.autolabel import validate_labels


def test_valid_unknown_sides_and_holdout():
    assert validate_labels({'sides':{'front':0,'left':1,'back':None}},[0,1,2])['sides']=={'front':0,'left':1}

@pytest.mark.parametrize('sides',[{'front':9},{'front':0,'left':0},{'invented':0}])
def test_invalid_vlm_output_rejected(sides):
    with pytest.raises(ValueError):validate_labels({'sides':sides},[0,1,2])


def test_all_photos_labeled_keeps_front_and_reserves_holdout():
    result=validate_labels({'sides':{'front':9,'left':6,'back':11}},[6,9,11])
    assert result['sides']=={'front':9,'left':6}
    assert '12' in result['note'] and 'проверки' in result['note']


def test_empty_labels_explain_manual_action():
    assert 'вручную' in validate_labels({'sides':{}},[0,1,2])['note']
