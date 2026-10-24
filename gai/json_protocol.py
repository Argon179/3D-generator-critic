"""Read a single complete root object, never salvage a child of truncated JSON."""
import json
from pathlib import Path


class ModelResponseError(ValueError):
    pass


def parse_response(text):
    positions=[i for c in ('{','[') if (i:=text.find(c))>=0]
    if not positions or text[min(positions)]!='{':
        raise ModelResponseError('Ответ должен содержать полный JSON-объект верхнего уровня')
    try:
        data,_=json.JSONDecoder().raw_decode(text[min(positions):])
    except json.JSONDecodeError as exc:
        raise ModelResponseError('JSON оборван или некорректен; вложенные объекты не используются') from exc
    if not isinstance(data,dict):raise ModelResponseError('Ожидался JSON-объект')
    return data


def request_json(generate,prompt,paths,stem,out,tokens,validate=lambda d:d):
    """Bounded retry, keep both raw responses and diagnostics for inspection."""
    out=Path(out);errors=[]
    for attempt in range(2):
        current=prompt
        if attempt:
            current+='\nYour previous response was invalid. Return ONE COMPLETE compact JSON object only. Do not repeat input observations. Validation error: '+errors[-1][:1500]
        raw=generate(current,paths,tokens)
        prefix=out/f'{stem}_attempt_{attempt+1}'
        prefix.with_suffix('.txt').write_text(raw,encoding='utf-8')
        (out/(stem+'_raw.txt')).write_text(raw,encoding='utf-8')
        (out/(stem+'_prompt.txt')).write_text(current,encoding='utf-8')
        try:
            result=validate(parse_response(raw))
        except (ValueError,TypeError,KeyError) as exc:
            detail=json.dumps(exc.errors(include_input=False,include_url=False),ensure_ascii=False,default=str) if hasattr(exc,'errors') else str(exc)
            errors.append(detail)
            (out/(stem+'_errors.json')).write_text(json.dumps(errors,ensure_ascii=False,indent=2))
            continue
        (out/(stem+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
        return result
    raise ModelResponseError('VLM дважды вернула неполный или недопустимый JSON; см. '+stem+'_errors.json')
