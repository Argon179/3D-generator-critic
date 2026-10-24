"""Validated, executable plans. Image boxes use the original render frame."""
from typing import Literal
import math
from pydantic import BaseModel, Field, model_validator


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
    operation: Literal['none','local_scale','local_smooth','remove_region','repair_mesh','regenerate'] = 'none'
    reason_ru: str = Field(default='',max_length=1500)
    regions: list[Region] = Field(default_factory=list,max_length=12)
    scale: float = Field(default=1,ge=.92,le=1.08)
    axis: Literal['x','y','z','all'] = 'all'

    @model_validator(mode='after')
    def independent_regions(self):
        ids=[r.reference_index for r in self.regions]
        if len(ids)!=len(set(ids)):
            raise ValueError('Only one region per reference')
        return self


def compatible_plan(edit_type,operation):
    if operation=='none':return True
    return operation in {
        'none':{'none'}, 'reshape':{'local_scale','local_smooth'},
        'remove':{'remove_region'}, 'repair':{'repair_mesh'},
        'add':{'regenerate'},
    }.get(edit_type,set())
