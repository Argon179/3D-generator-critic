"""More exact known views and rigid orientation alignment for full reconstruction."""
import numpy as np


def plan_regeneration(job,refs,metrics):
    """Never invent unknown camera angles; preserve at least one untouched reference."""
    selected=dict(job.get('selected',{})) if isinstance(job.get('selected'),dict) else job.get('selected')
    info={'selected':selected,'yaw_degrees':0,'input_views_changed':False}
    if job.get('backend')!='hunyuan' or not isinstance(selected,dict):return info
    if len(refs)<5 or any(r.get('azimuth_deg') is None for r in refs):return info
    pairs={p['reference_index']:p for p in metrics['pairs']}
    offsets=[(pairs[i]['azimuth_deg']-float(r['azimuth_deg']))%360 for i,r in enumerate(refs)]
    if any(abs((a-offsets[0]+180)%360-180)>1 for a in offsets):return info
    angles={round(float(r['azimuth_deg'])%360,3):i for i,r in enumerate(refs)}
    candidates=[];origin=float(refs[selected['front']]['azimuth_deg'])
    for angle,index in angles.items():
        wanted=[(angle+delta)%360 for delta in [0,90,180,270]]
        if not all(round(a,3) in angles for a in wanted):continue
        proposed=dict(zip(['front','left','back','right'],[angles[round(a,3)] for a in wanted]))
        if len(set(selected.values())|set(proposed.values()))>=len(refs):continue
        distance=abs((angle-origin+180)%360-180)
        candidates.append((distance,proposed))
    if candidates and len(selected)<4:
        proposed=min(candidates,key=lambda x:x[0])[1]
        # The new front is native camera azimuth 0; align it with the baseline frame.
        yaw=float(pairs[proposed['front']]['azimuth_deg'])
        info.update(selected=proposed,yaw_degrees=yaw,input_views_changed=True)
    return info


def register_translation(mesh,source,generation_views):
    """Rigid centering using conditioning photos only; control views are never fitted."""
    import json
    from pathlib import Path
    from gai.local_editor import raster_faces
    from gai.shape_evidence import foreground
    path=Path(source)/'shape_evidence/catalog.json'
    if not path.is_file():return [0,0,0]
    catalog=json.loads(path.read_text());rows=[];rhs=[]
    for record in catalog['views']:
        if record['reference_index'] not in generation_views:continue
        matrix=np.asarray(record['projection_matrix'])
        predicted=raster_faces(mesh,matrix)>=0;_,target=foreground(record['target'])
        py,px=np.nonzero(predicted);ty,tx=np.nonzero(target)
        if not len(px) or not len(tx):continue
        difference=np.array([tx.mean()-px.mean(),ty.mean()-py.mean()])/192
        rows.extend([matrix[0,:3]*.5,-matrix[1,:3]*.5]);rhs.extend(difference)
    if len(rows)<4:return [0,0,0]
    shift=np.linalg.lstsq(np.asarray(rows),np.asarray(rhs),rcond=None)[0]
    limit=.25*np.ptp(mesh.vertices,axis=0).max()
    if np.linalg.norm(shift)>limit:return [0,0,0]
    mesh.apply_translation(shift)
    return shift.tolist()
