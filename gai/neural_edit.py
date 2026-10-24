"""Validated mesh-conditioned diffusion edit jobs; no neural imports in the server."""
from pathlib import Path
import hashlib
import json
import numpy as np
from gai.local_editor import UnsupportedEdit


def prepare_neural_edit(mesh_path,source,proposal_id):
    catalog=json.loads((Path(source)/'shape_evidence/catalog.json').read_text())
    if catalog.get('mesh_sha256')!=hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest():
        raise UnsupportedEdit('Neural edit proposal belongs to another mesh')
    proposal=next((p for p in catalog['proposals'] if p['id']==proposal_id),None)
    if proposal is None:raise UnsupportedEdit('Unknown neural edit proposal')
    views={v['reference_index']:v for v in catalog['views']}
    indexes={r['reference_index'] for r in proposal['regions']}
    if len(indexes)<2 or len(indexes)>=len(views) or not indexes<=views.keys():
        raise UnsupportedEdit('Neural editing requires two training views and an untouched view')
    if catalog.get('reserved_control') in indexes:
        raise UnsupportedEdit('Reserved control photo cannot supervise a neural edit')
    return catalog,proposal,[views[i] for i in sorted(indexes)]


def deformation_weights(mesh,indexes):
    """Fixed exterior and boundary; feather motion into the selected volume."""
    indexes=np.asarray(indexes,dtype=int)
    if len(indexes)<8 or len(np.unique(indexes))!=len(indexes) or indexes.min()<0 or indexes.max()>=len(mesh.vertices):
        raise UnsupportedEdit('Invalid neural editing vertex selection')
    if len(indexes)>.35*len(mesh.vertices):raise UnsupportedEdit('Neural edit region exceeds 35% of the mesh')
    selected=np.zeros(len(mesh.vertices),bool);selected[indexes]=True
    distances=np.full(len(selected),5,dtype=int);distances[~selected]=0
    for _ in range(4):
        old=distances.copy()
        for i in indexes:
            neighbors=mesh.vertex_neighbors[i]
            if len(neighbors):distances[i]=min(old[i],old[neighbors].min()+1)
    weights=np.where(selected,np.clip((distances-1)/3,0,1),0).astype(np.float32)
    if not np.any(weights):raise UnsupportedEdit('No interior vertices in the neural edit region')
    return weights


def editor_config(cfg,editor):
    result=dict(cfg)
    if editor not in {'standard','diffusion'}:raise ValueError('Unknown mesh editor')
    result['neural_editor_enabled']=editor=='diffusion'
    if editor=='diffusion':
        from gai.core import ROOT
        cache=ROOT/'.cache/huggingface/hub'/('models--'+cfg['neural_editor_model'].replace('/','--'))/'snapshots'/cfg['neural_editor_revision']
        required=['unet/diffusion_pytorch_model.fp16.safetensors','vae/diffusion_pytorch_model.fp16.safetensors','text_encoder/model.fp16.safetensors','tokenizer/vocab.json','scheduler/scheduler_config.json']
        if not all((cache/name).is_file() for name in required):
            raise ValueError('Neural editor weights are missing. Run scripts/setup_neural_editor.py first.')
    return result
