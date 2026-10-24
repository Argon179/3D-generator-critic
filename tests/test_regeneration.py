from gai.regeneration import plan_regeneration


def test_more_known_views_leave_a_real_holdout_and_register_to_baseline():
    refs=[{'azimuth_deg':a} for a in [330,0,90,150,180,270,300]]
    metrics={'pairs':[{'reference_index':i,'azimuth_deg':(r['azimuth_deg']+30)%360} for i,r in enumerate(refs)]}
    plan=plan_regeneration({'backend':'hunyuan','selected':{'front':0,'back':3}},refs,metrics)
    assert plan['selected']=={'front':1,'left':2,'back':4,'right':5}
    assert plan['yaw_degrees']==30 and plan['input_views_changed']
    assert set(range(7)) - ({0,3}|set(plan['selected'].values()))=={6}


def test_unknown_or_inconsistent_angles_never_invent_extra_views():
    selected={'front':0,'back':2};job={'backend':'hunyuan','selected':selected}
    assert plan_regeneration(job,[{'azimuth_deg':None}]*6,{'pairs':[]})['selected']==selected
    refs=[{'azimuth_deg':a} for a in [0,90,180,270,300]]
    metrics={'pairs':[{'reference_index':i,'azimuth_deg':(-r['azimuth_deg'])%360} for i,r in enumerate(refs)]}
    assert not plan_regeneration(job,refs,metrics)['input_views_changed']
