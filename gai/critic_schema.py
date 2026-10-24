"""Validated, executable plans. Image boxes use the original render frame."""
from typing import Literal
import math
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Region(BaseModel):
    reference_index: int = Field(ge=0)
    bbox: tuple[float, float, float, float]

    @model_validator(mode='after')
    def ordered_box(self):
        x0,y0,x1,y1=self.bbox
        if not all(math.isfinite(v) and 0<=v<=1 for v in self.bbox) or x0>=x1 or y0>=y1:
            raise ValueError('bbox must be normalized [left, top, right, bottom]')
        return self


class EditPlan(BaseModel):
    operation: Literal['none','local_scale','local_smooth','remove_region','repair_mesh','regenerate','fit_silhouette'] = 'none'
    proposal_id: str|None = Field(default=None,max_length=80)
    evidence_basis: Literal['vlm_boxes','silhouette_catalog']='vlm_boxes'
    reason_en: str|None = None
    reason_ru: str = Field(default='',max_length=1500)
    regions: list[Region] = Field(default_factory=list,max_length=12)
    scale: float = Field(default=1,ge=.92,le=1.08)
    axis: Literal['x','y','z','all'] = 'all'

    @model_validator(mode='after')
    def independent_regions(self):
        if self.operation=='fit_silhouette' and (self.evidence_basis!='silhouette_catalog' or not self.proposal_id):
            raise ValueError('Contour fitting requires a compiled proposal ID')
        if self.evidence_basis=='silhouette_catalog' and self.operation not in {'fit_silhouette','local_smooth','regenerate','none'}:
            raise ValueError('Catalog evidence cannot authorize free-form mesh operations')
        ids=[r.reference_index for r in self.regions]
        if len(ids)!=len(set(ids)):
            raise ValueError('Only one region per reference')
        return self


def compatible_plan(edit_type,operation):
    if operation=='none':return True
    return operation in {
        'none':{'none'}, 'reshape':{'local_scale','local_smooth','fit_silhouette','regenerate'},
        'remove':{'remove_region'}, 'repair':{'repair_mesh'},
        'add':{'regenerate'},
    }.get(edit_type,set())


class Critique(BaseModel):
    model_config=ConfigDict(extra='ignore')
    text_language: str|None=None
    translation_status: str|None=None
    summary_en: str|None=None
    summary_ru: str=Field(max_length=2000)
    camera_mismatch: bool
    confidence: float=Field(ge=0,le=1)
    edit_type: Literal["none","add","remove","reshape","repair"]|None=None
    edit_plan: EditPlan|None=None
    view_assessments: list[dict]=Field(default_factory=list,max_length=24)
    should_edit: bool
    issues: list[dict]=Field(default_factory=list,max_length=10)
    quality_warnings_en: list[str]=Field(default_factory=list,max_length=3)
    quality_warnings: list[str]=Field(default_factory=list,max_length=3)
    base_description: str=Field(min_length=3,max_length=250)
    target_description: str=Field(min_length=3,max_length=250)

    @field_validator('base_description','target_description')
    @classmethod
    def short_clip_prompt(cls,v):
        # CLIP has a 77-token context; conservative character bound is also checked with CLIP by worker.
        if '\n' in v:raise ValueError('Описание должно быть одной строкой')
        return v.strip()

    @model_validator(mode='after')
    def consistent_edit(self):
        if self.camera_mismatch or not self.issues or self.edit_type=='none':
            self.should_edit=False
        if not self.should_edit:
            self.edit_type='none'
            self.target_description=self.base_description
        if self.edit_plan is not None and self.edit_plan.operation!='none' and (not self.should_edit or not compatible_plan(self.edit_type,self.edit_plan.operation)):
            self.edit_plan=EditPlan(reason_ru='Нет согласованного исполнимого плана')
        if self.edit_plan and self.edit_plan.regions and self.view_assessments and self.edit_plan.evidence_basis!='silhouette_catalog':
            observations={v.get('reference_index'):v for v in self.view_assessments}
            for region in self.edit_plan.regions:
                observation=observations.get(region.reference_index,{})
                boxes=[d.get('render_bbox') for d in observation.get('defects',[]) if isinstance(d,dict)]
                supported=any(isinstance(b,(list,tuple)) and len(b)==4 and all(isinstance(x,(int,float)) and abs(x-y)<=.03 for x,y in zip(b,region.bbox)) for b in boxes)
                if observation.get('camera_mismatch') is not False or not supported:
                    self.edit_plan=EditPlan(reason_ru='Область не подтверждена наблюдением по этому фото')
                    break
        return self



class ViewDefect(BaseModel):
    part: str
    kind: Literal['extra','missing','shape','topology','appearance']
    observation_ru: str = Field(max_length=1500)
    render_bbox: tuple[float,float,float,float]
    confidence: float = Field(ge=0,le=1)

    @model_validator(mode='after')
    def valid_box(self):
        Region(reference_index=0,bbox=self.render_bbox)
        return self


class ViewAssessment(BaseModel):
    reference_index: int=Field(ge=0)
    camera_mismatch: bool
    observation_ru: str=Field(min_length=1,max_length=2000)
    defects: list[ViewDefect]=Field(default_factory=list,max_length=6)


def validate_critic_report(data,base_description,observations):
    data=dict(data,base_description=base_description,view_assessments=observations)
    # Core assessment errors require a retry. An invalid executable plan merely
    # disables execution; it must not destroy an otherwise valid assessment.
    try:
        if 'edit_plan' not in data:raise ValueError('No executable plan')
        data['edit_plan']=EditPlan.model_validate(data['edit_plan']).model_dump()
    except (ValueError,TypeError):
        data['edit_plan']=EditPlan(reason_ru='Оценка получена, но VLM не сформировала допустимый план правки').model_dump()
        data['quality_warnings']=['Оценка получена; автоматическая правка недоступна из-за формата плана.']
    return Critique.model_validate(data).model_dump()


class ProposalChoice(BaseModel):
    action: Literal['fit_silhouette','local_smooth','regenerate','none']
    proposal_id: str|None=None
    camera_mismatch: bool
    confidence: float=Field(ge=0,le=1)
    reason_ru: str=Field(min_length=3,max_length=1500)


def compile_choice(data,catalog,base_description,observations):
    choice=ProposalChoice.model_validate(data)
    proposals={p['id']:p for p in catalog['proposals']}
    if choice.action!='none' and choice.proposal_id not in proposals:
        raise ValueError('Select an existing proposal_id, never invent coordinates')
    active=choice.action in {'fit_silhouette','local_smooth','regenerate'} and not choice.camera_mismatch and choice.confidence>=.65
    if active:
        p=proposals[choice.proposal_id]
        if choice.action not in p.get('permitted_operations',['fit_silhouette','local_smooth','regenerate']):raise ValueError('This operation was excluded after an unsuccessful trial')
        plan=EditPlan(operation=choice.action,proposal_id=p['id'],evidence_basis='silhouette_catalog',
                      regions=p['regions'],reason_ru=choice.reason_ru)
    else:plan=EditPlan(reason_ru=choice.reason_ru)
    return Critique(summary_ru=choice.reason_ru,camera_mismatch=choice.camera_mismatch,confidence=choice.confidence,
        should_edit=active,edit_type='reshape' if active else 'none',edit_plan=plan,
        issues=[{'part':'selected contour region','observation_ru':choice.reason_ru,'suggestion_ru':('Reconstruct from the original photos and check all views.' if choice.action=='regenerate' else 'Apply the selected local operation, then check the remaining views.')}] if active else [],
        base_description=base_description,target_description=base_description,view_assessments=observations).model_dump(mode='json')
