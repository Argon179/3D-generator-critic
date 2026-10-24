"""Strict validation of independently generated before/after assessments."""
from pydantic import BaseModel,StrictBool,Field
from typing import Literal


class BlindReview(BaseModel):
    preferred:Literal['A','B','tie']
    camera_mismatch:StrictBool
    a_has_unique_defect:StrictBool
    b_has_unique_defect:StrictBool
    confidence:float=Field(ge=0,le=1)
    explanation_en:str|None=None
    explanation_ru:str=Field(min_length=3,max_length=2000)


def unblind_review(data,index,after_label):
    review=BlindReview.model_validate(data)
    return ViewReview(reference_index=index,camera_mismatch=review.camera_mismatch,
        improved=review.preferred==after_label,
        worsened=review.preferred not in {after_label,'tie'} or getattr(review,after_label.lower()+'_has_unique_defect'),
        confidence=review.confidence,explanation_ru=review.explanation_ru,explanation_en=review.explanation_en).model_dump()


class ViewReview(BaseModel):
    reference_index:int=Field(ge=0)
    camera_mismatch:StrictBool
    improved:StrictBool
    worsened:StrictBool
    confidence:float=Field(ge=0,le=1)
    explanation_en:str|None=None
    assessment_valid:bool=True
    inconclusive:bool=False
    explanation_ru:str=Field(max_length=2000)


def summarize_review(assessments,evidence_indexes):
    views=[ViewReview.model_validate(x) for x in assessments]
    ids=[v.reference_index for v in views]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate verification view')
    evidence=set(evidence_indexes)
    improved=[v.reference_index for v in views if v.reference_index in evidence and v.improved and not v.camera_mismatch and v.confidence>=.65 and not v.worsened]
    # A reported new defect blocks acceptance even when confidence is low.
    clean=not any(v.worsened or v.camera_mismatch or not v.assessment_valid for v in views)
    accepted=clean and evidence.issubset(ids) and len(improved)>=2
    return {'accepted':accepted,'improved_views':improved,'incomplete_views':[v.reference_index for v in views if not v.assessment_valid],
            'reason_en':('Review incomplete: invalid responses for one or more views.' if any(not v.assessment_valid for v in views) else 'Independent VLM confirmed improvement in at least two views without new defects.' if accepted else 'Independent VLM did not confirm improvement or detected a new discrepancy.'),'views':[v.model_dump() for v in views],
            'reason_ru':'Независимая VLM подтвердила улучшение минимум в двух видах без новых дефектов' if accepted else 'Независимая VLM не подтвердила улучшение либо обнаружила новое расхождение'}


def verification_context(proposal):
    """Do not prime the verifier with the critic's verdict, confidence or rationale."""
    plan=proposal.get('edit_plan') or {}
    return {'operation':plan.get('operation'),'reference_indexes':[r['reference_index'] for r in plan.get('regions',[])],
            'task':'Compare actual visible shape before/after against reference; no improvement is assumed.'}


def consistent_review(first,second,index):
    """Compare the same geometry in both column orders; expose positional bias."""
    a=ViewReview.model_validate(first);b=ViewReview.model_validate(second)
    complete=a.assessment_valid and b.assessment_valid
    confidence=min(a.confidence,b.confidence)
    same=(a.improved,a.worsened,a.camera_mismatch)==(b.improved,b.worsened,b.camera_mismatch)
    return ViewReview(reference_index=index,assessment_valid=complete,
        improved=complete and same and a.improved and confidence>=.65,
        worsened=complete and same and a.worsened and confidence>=.65,
        camera_mismatch=complete and a.camera_mismatch and b.camera_mismatch,
        confidence=confidence,explanation_en='Order 1: '+(a.explanation_en or a.explanation_ru)+' Order 2: '+(b.explanation_en or b.explanation_ru),
        explanation_ru='Order 1: '+(a.explanation_en or a.explanation_ru)+' Order 2: '+(b.explanation_en or b.explanation_ru),
        inconclusive=not complete or not same or confidence<.65).model_dump()


def reconstruction_evidence(before,after,review,margin=.005):
    """Narrow fallback for independently improved reconstructions, never local fits.

    An order-sensitive VLM cannot veto geometry on its own. Require several
    independent numerical checks and a structurally valid mesh instead.
    """
    views=review.get('views',[])
    if not views or any(not v.get('assessment_valid',True) or v['camera_mismatch'] for v in views):return False
    # A whole-object veto needs corroboration in distinct views, not just swapped columns.
    suspected=[v['reference_index'] for v in views if v['worsened']]
    if len(suspected)>=2:return False
    expected={p['reference_index'] for p in after['pairs']}
    if {v['reference_index'] for v in views}!=expected:return False
    controls=[p['reference_index'] for p in before['pairs'] if not p.get('used_for_generation',False) and not p.get('used_for_edit',False)]
    if not controls:return False
    a={p['reference_index']:p for p in before['pairs']};b={p['reference_index']:p for p in after['pairs']}
    required=['iou','boundary_f1','contour_distance']
    if any(k not in a[i] or k not in b[i] for i in expected for k in required):return False
    import numpy as np
    if np.mean([b[i]['iou']-a[i]['iou'] for i in controls])<margin:return False
    if np.mean([b[i]['boundary_f1']-a[i]['boundary_f1'] for i in controls])<.02:return False
    if min(b[i]['boundary_f1']-a[i]['boundary_f1'] for i in controls)<-.005:return False
    if np.mean([b[i]['contour_distance']-a[i]['contour_distance'] for i in controls])>0:return False
    if min(b[i]['iou']-a[i]['iou'] for i in expected)<-.015:return False
    old=before.get('geometry',{});new=after.get('geometry',{})
    if old.get('euler_number') is None or old.get('euler_number')!=new.get('euler_number'):return False
    return new.get('watertight') is True and new.get('winding_consistent') is True and (new.get('volume') or 0)>0 and new.get('components')==1 and old.get('components')==1
