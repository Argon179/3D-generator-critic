import pytest
from gai.action_choice import action_labels,decision_data


def test_only_feasible_actions_are_decodable_and_no_change_is_always_possible():
    labels=action_labels(['local_smooth'])
    assert labels=={'B':'local_smooth','N':'none','C':'camera_mismatch'}
    with pytest.raises(ValueError):decision_data('A',{'A':.9},'region_01',['local_smooth'])
    assert decision_data('B',{'B':.8},'region_01',['local_smooth'])['action']=='local_smooth'
    assert decision_data('N',{'N':.9},'region_01',[])['action']=='none'
    assert decision_data('C',{'C':.9},'region_01',[])['camera_mismatch']


def test_whole_object_evidence_spreads_camera_angles():
    from gai.action_choice import global_pairs
    pairs=[{'reference_index':i,'camera':{'azimuth_deg':a}} for i,a in enumerate([0,30,120,180,210,300,330])]
    chosen=global_pairs(pairs)
    assert [p['reference_index'] for p in chosen]==[0,3,2]
    assert global_pairs([])==[]
    assert len(global_pairs([{'reference_index':0},{'reference_index':1}]))==2


def test_global_reconstruction_compiles_full_views_without_local_vertex_selection():
    from gai.critic_schema import compile_choice
    proposal={'id':'global_reconstruction','permitted_operations':['regenerate'],
              'regions':[{'reference_index':i,'bbox':[0,0,1,1]} for i in [0,2,3]]}
    decision=decision_data('R',{'R':.9},proposal['id'],['regenerate'])
    result=compile_choice(decision,{'proposals':[proposal]},'a technical part',[])
    assert result['should_edit'] and result['edit_plan']['operation']=='regenerate'
    assert len(result['edit_plan']['regions'])==3
