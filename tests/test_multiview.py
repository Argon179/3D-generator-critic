import pytest
from gai.multiview import select_views


def test_hunyuan_selects_real_cardinal_views_and_reserves_holdout():
    refs=[{'azimuth_deg':i*30} for i in range(12)]
    selected=select_views(refs,'hunyuan')
    assert selected=={'front':0,'left':3,'back':6,'right':9}
    assert len(set(range(12))-set(selected.values()))==8


def test_hunyuan_rejects_repeated_or_missing_sides():
    refs=[{}]*5
    with pytest.raises(ValueError):select_views(refs,'hunyuan',{'front':0,'left':0})
    with pytest.raises(ValueError):select_views(refs,'hunyuan',{'left':1,'back':2})
    with pytest.raises(ValueError):select_views(refs,'hunyuan')
    with pytest.raises(ValueError):select_views([{}]*2,'hunyuan',{'front':0,'back':1})


def test_da3_keeps_real_holdout_and_first_image():
    refs=[{}]*12
    chosen=select_views(refs,'da3')
    assert len(chosen)==9 and 0 in chosen
    assert set(chosen).isdisjoint(set(range(12))-set(chosen))
    with pytest.raises(ValueError):select_views([{},{}],'da3')


def test_cached_weights_require_complete_files_and_respect_revision(tmp_path):
    from gai.multiview import cached_snapshot
    base=tmp_path/'hub/models--author--model'
    partial=base/'snapshots/partial';partial.mkdir(parents=True)
    (partial/'config.json').write_text('{}')
    assert cached_snapshot(tmp_path,'author/model',['config.json','model.safetensors']) is None
    complete=base/'snapshots/complete';complete.mkdir()
    (complete/'config.json').write_text('{}');(complete/'model.safetensors').write_bytes(b'test')
    assert cached_snapshot(tmp_path,'author/model',['config.json','model.safetensors'])==complete
    assert cached_snapshot(tmp_path,'author/model',['config.json','model.safetensors'],revision='partial') is None
