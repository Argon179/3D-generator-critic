"""Strict validation of independently generated before/after assessments."""
from pydantic import BaseModel,StrictBool,Field


class ViewReview(BaseModel):
    reference_index:int=Field(ge=0)
    camera_mismatch:StrictBool
    improved:StrictBool
    worsened:StrictBool
    confidence:float=Field(ge=0,le=1)
    explanation_ru:str=Field(max_length=2000)


def summarize_review(assessments,evidence_indexes):
    views=[ViewReview.model_validate(x) for x in assessments]
    ids=[v.reference_index for v in views]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate verification view')
    evidence=set(evidence_indexes)
    improved=[v.reference_index for v in views if v.reference_index in evidence and v.improved and not v.camera_mismatch and v.confidence>=.65 and not v.worsened]
    # A reported new defect blocks acceptance even when confidence is low.
    clean=not any(v.worsened or v.camera_mismatch for v in views)
    accepted=clean and evidence.issubset(ids) and len(improved)>=2
    return {'accepted':accepted,'improved_views':improved,'views':[v.model_dump() for v in views],
            'reason_ru':'Независимая VLM подтвердила улучшение минимум в двух видах без новых дефектов' if accepted else 'Независимая VLM не подтвердила улучшение либо обнаружила новое расхождение'}


def verification_context(proposal):
    """Do not prime the verifier with the critic's verdict, confidence or rationale."""
    plan=proposal.get('edit_plan') or {}
    return {'operation':plan.get('operation'),'reference_indexes':[r['reference_index'] for r in plan.get('regions',[])],
            'task':'Compare actual visible shape before/after against reference; no improvement is assumed.'}
