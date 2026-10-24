import json
import pytest
from gai.json_protocol import parse_response,request_json,ModelResponseError
from gai.critic_schema import Critique


def report():
    return dict(summary_ru='Test',camera_mismatch=False,confidence=.8,should_edit=False,
                base_description='a cat',target_description='a cat',issues=[])


def test_truncated_root_never_becomes_a_valid_nested_issue():
    raw='{"summary_ru":"Test", "issues":[{"part":"head","observation_ru":"extra"}], "unfinished":"'
    with pytest.raises(ModelResponseError,match='оборван'):parse_response(raw)
    assert parse_response('```json\n'+json.dumps(report())+'\n```')==report()
    with pytest.raises(ModelResponseError):parse_response('[{"part":"head"}]')


def test_full_schema_is_validated_and_retried(tmp_path):
    replies=iter([json.dumps({'part':'head'}),json.dumps(report())]);prompts=[]
    def generate(prompt,paths,tokens):prompts.append(prompt);return next(replies)
    data=request_json(generate,'test',[],'plan',tmp_path,100,lambda d:Critique.model_validate(d).model_dump())
    assert data['summary_ru']=='Test' and len(prompts)==2
    assert 'previous response was invalid' in prompts[1]
    assert (tmp_path/'plan_attempt_1.txt').exists() and (tmp_path/'plan_errors.json').exists()


def test_two_bad_responses_are_explicitly_rejected(tmp_path):
    calls=[]
    def generate(*args):calls.append(1);return '{"issues":[{"part":"head"}],"unfinished":'
    with pytest.raises(ModelResponseError):request_json(generate,'test',[],'plan',tmp_path,100)
    assert len(calls)==2


def test_invalid_optional_plan_does_not_discard_valid_assessment():
    from gai.critic_schema import validate_critic_report
    data=dict(report(),should_edit=True,edit_type='remove',issues=[{'part':'extra protrusion'}],
              edit_plan={'operation':'remove_region','regions':[{'reference_index':0,'bbox':[.1,.1,.1,.2,.2,.2]}]})
    result=validate_critic_report(data,'a cat',[])
    assert result['summary_ru']=='Test' and result['issues']
    assert result['edit_plan']['operation']=='none' and result['quality_warnings']


def test_missing_core_fields_are_not_filled_with_invented_judgment():
    from gai.critic_schema import validate_critic_report
    with pytest.raises(ValueError):validate_critic_report({'part':'head'},'a cat',[])


def test_validator_exception_context_is_logged_without_breaking_retry(tmp_path):
    from gai.critic_schema import Region
    replies=iter([json.dumps({'reference_index':0,'bbox':[.7,.7,.2,.2]}),json.dumps({'reference_index':0,'bbox':[.1,.1,.2,.2]})])
    result=request_json(lambda *args:next(replies),'test',[],'region',tmp_path,100,
                        lambda d:Region.model_validate(d).model_dump())
    assert result['bbox']==(.1,.1,.2,.2)
    assert 'bbox must be normalized' in (tmp_path/'region_errors.json').read_text()
