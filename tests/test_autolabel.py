import pytest
from gai.autolabel import validate_labels


def test_valid_unknown_sides_and_holdout():
    assert validate_labels({'sides':{'front':0,'left':1,'back':None}},[0,1,2])['sides']=={'front':0,'left':1}

@pytest.mark.parametrize('sides',[{'front':9},{'front':0,'left':0},{'front':0,'left':1,'back':2},{'invented':0}])
def test_invalid_vlm_output_rejected(sides):
    with pytest.raises(ValueError):validate_labels({'sides':sides},[0,1,2])
