from gai.localization import bilingual_report,text_pairs


def test_translation_only_changes_text_and_preserves_english_source_and_controls():
    source={'summary_ru':'An extra surface.','should_edit':True,'confidence':.8,'edit_plan':{'operation':'regenerate','regions':[{'bbox':[0,0,1,1]}],'reason_ru':'Rebuild the shape.'},'views':[{'explanation_ru':'Better contour.','improved':True}]}
    translations={'An extra surface.':'Лишняя поверхность.','Rebuild the shape.':'Перестроить форму.','Better contour.':'Более точный контур.'}
    result=bilingual_report(source,lambda texts,s,t:[translations[x] for x in texts])
    assert result['summary_en']==source['summary_ru'] and result['summary_ru']=='Лишняя поверхность.'
    assert result['edit_plan']['operation']=='regenerate' and result['edit_plan']['regions']==source['edit_plan']['regions']
    assert result['should_edit'] is True and result['confidence']==.8 and result['views'][0]['improved'] is True
    assert source['summary_ru']=='An extra surface.'
    assert text_pairs(result)['Более точный контур.']=='Better contour.'


def test_legacy_russian_is_preserved_and_only_translated_for_english_display():
    result=bilingual_report({'summary_ru':'Лишняя поверхность.'},lambda texts,s,t:['An extra surface.'])
    assert result['summary_ru']=='Лишняя поверхность.' and result['summary_en']=='An extra surface.'


def test_incomplete_visual_response_is_not_a_camera_mismatch_or_a_valid_acceptance():
    from gai.review import summarize_review
    views=[dict(reference_index=i,camera_mismatch=False,improved=i<2,worsened=False,confidence=.9,explanation_ru='Visible change.',assessment_valid=i!=2) for i in range(3)]
    result=summarize_review(views,[0,1])
    assert not result['accepted'] and result['incomplete_views']==[2]
    assert all(not v['camera_mismatch'] for v in result['views'])


def test_swapping_columns_exposes_position_bias_and_preserves_english_explanations():
    from gai.review import consistent_review
    a=dict(reference_index=0,camera_mismatch=False,improved=True,worsened=False,confidence=.9,explanation_ru='B is better.')
    b=dict(a,improved=False,worsened=True)
    result=consistent_review(a,b,0)
    assert result['inconclusive'] and not result['improved'] and not result['worsened']
    assert result['explanation_en'].startswith('Order 1:')


def test_uncertain_vlm_requires_multiple_independent_checks_for_reconstruction():
    from gai.review import reconstruction_evidence
    import copy
    before={'geometry':{'components':1,'watertight':True,'euler_number':2,'winding_consistent':True,'volume':1},'pairs':[{'reference_index':i,'used_for_generation':i==0,'iou':.8,'boundary_f1':.5,'contour_distance':.02} for i in range(3)]}
    after=copy.deepcopy(before)
    for p in after['pairs']:p.update(iou=.84,boundary_f1=.6,contour_distance=.015)
    review={'views':[{'reference_index':i,'assessment_valid':True,'worsened':False,'camera_mismatch':False,'inconclusive':True} for i in range(3)]}
    assert reconstruction_evidence(before,after,review)
    review['views'][1]['worsened']=True;review['views'][2]['worsened']=True
    assert not reconstruction_evidence(before,after,review)
    review['views'][1]['worsened']=False;review['views'][2]['worsened']=False;after['pairs'][1]['boundary_f1']=.49
    assert not reconstruction_evidence(before,after,review)
    after['geometry']['watertight']=False
    assert not reconstruction_evidence(before,after,review)


def test_bilingual_assessment_survives_pipeline_schema_roundtrip():
    from gai.critic_schema import Critique
    data={'summary_ru':'No edit needed.','camera_mismatch':False,'confidence':.8,'should_edit':False,'issues':[],
          'base_description':'a part','target_description':'a part','quality_warnings':['View is ambiguous.']}
    phrases={'No edit needed.':'Правка не нужна.','View is ambiguous.':'Неоднозначный ракурс.'}
    bilingual=bilingual_report(data,lambda texts,s,t:[phrases[x] for x in texts])
    saved=Critique.model_validate(bilingual).model_dump()
    assert saved['summary_en']=='No edit needed.' and saved['summary_ru']=='Правка не нужна.'
    assert text_pairs(saved)['Неоднозначный ракурс.']=='View is ambiguous.'
    assert saved['quality_warnings_en']==['View is ambiguous.'] and saved['quality_warnings']==['Неоднозначный ракурс.']
