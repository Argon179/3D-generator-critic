"""Coarse silhouette alignment. Never treat this as 3D ground truth."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage


def normalized_mask(path, size=128):
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im)
    alpha = a[:, :, 3]
    if alpha.min() == alpha.max():
        raise ValueError(f"Нет маски прозрачности: {path}")
    mask = alpha > 127
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"Пустая маска: {path}")
    # Translation and scale are intentionally removed. Measures contour shape only.
    crop = Image.fromarray(mask[ys.min():ys.max()+1, xs.min():xs.max()+1].astype('uint8') * 255)
    factor = (size - 12) / max(crop.size)
    crop = crop.resize((max(1,round(crop.width*factor)),max(1,round(crop.height*factor))), Image.Resampling.NEAREST)
    canvas = Image.new("L", (size, size))
    canvas.paste(crop, ((size - crop.width)//2, (size - crop.height)//2))
    return np.asarray(canvas) > 127


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def contour_metrics(a,b):
    edge_a=a & ~ndimage.binary_erosion(a)
    edge_b=b & ~ndimage.binary_erosion(b)
    da=ndimage.distance_transform_edt(~edge_a)
    db=ndimage.distance_transform_edt(~edge_b)
    ab=db[edge_a];ba=da[edge_b]
    precision=float(np.mean(ab<=2));recall=float(np.mean(ba<=2))
    f1=2*precision*recall/max(precision+recall,1e-9)
    return {'boundary_f1':f1,'contour_distance':float((ab.mean()+ba.mean())/2/np.hypot(*a.shape))}


def align(references, render_dir, fixed_pairs=None):
    render_dir = Path(render_dir)
    all_views = json.loads((render_dir / 'views.json').read_text())
    views = [v for v in all_views if abs(v['elevation_deg']) != 90]
    render_masks = {v['file']: normalized_mask(render_dir / v['file']) for v in all_views}
    ref_masks = [normalized_mask(r['prepared']) for r in references]
    camera_training=[i for i,r in enumerate(references) if r.get('used_for_generation')]
    if not camera_training:camera_training=list(range(len(references)))
    if fixed_pairs is not None:
        pairs = [dict(p, iou=iou(ref_masks[p['reference_index']], render_masks[p['render']])) for p in fixed_pairs]
    else:
        # For datasets with known relative azimuths, use one shared offset/direction/elevation.
        known = all(r.get('azimuth_deg') is not None for r in references)
        pairs = None
        best = -1
        if known:
            by_angle = {(v['azimuth_deg'], v['elevation_deg']): v for v in views}
            for elev in sorted({v['elevation_deg'] for v in views}):
                for sign in (1, -1):
                    for offset in sorted({v['azimuth_deg'] for v in views}):
                        candidate = []
                        for k, r in enumerate(references):
                            angle = (offset + sign * r['azimuth_deg']) % 360
                            v = by_angle.get((angle, elev))
                            if v is None:
                                break
                            candidate.append({'reference_index': k, 'render': v['file'], 'azimuth_deg': angle, 'elevation_deg': elev, 'iou': iou(ref_masks[k], render_masks[v['file']])})
                        score = np.mean([p['iou'] for i,p in enumerate(candidate) if i in camera_training]) if len(candidate) == len(references) else -1
                        if score > best:
                            best, pairs = score, candidate
        labels=[r.get('view_label') for r in references]
        labeled=any(label in {'front','back','left','right','top','bottom'} for label in labels)
        if pairs is None and labeled:
            # One shared horizontal frame; top/bottom must never match a side silhouette.
            angles={'front':0,'right':90,'back':180,'left':270}
            candidates=[]
            for offset in (0,90,180,270):
                for sign in (1,-1):
                    candidate=[]
                    for k,label in enumerate(labels):
                        if label in angles:
                            wanted=(offset+sign*angles[label])%360
                            options=[v for v in views if v['elevation_deg']==0 and v['azimuth_deg']==wanted]
                        elif label in {'top','bottom'}:
                            options=[v for v in all_views if v['elevation_deg']==(90 if label=='top' else -90)]
                        elif label=='isometric':options=[v for v in views if v['elevation_deg']>0]
                        else:options=views
                        if not options:break
                        v=max(options,key=lambda v:iou(ref_masks[k],render_masks[v['file']]))
                        candidate.append({'reference_index':k,'render':v['file'],'azimuth_deg':v['azimuth_deg'],'elevation_deg':v['elevation_deg'],'iou':iou(ref_masks[k],render_masks[v['file']])})
                    if len(candidate)==len(references):
                        score=np.mean([candidate[i]['iou'] for i in camera_training])
                        candidates.append((score,candidate))
            if candidates:pairs=max(candidates,key=lambda x:x[0])[1]
        if pairs is None:
            # Unknown camera: independent nearest silhouette, explicitly low confidence.
            pairs = []
            for k, mask in enumerate(ref_masks):
                v = max(views, key=lambda v:iou(mask,render_masks[v['file']]))
                pairs.append({'reference_index': k, 'render': v['file'], 'azimuth_deg': v['azimuth_deg'], 'elevation_deg': v['elevation_deg'], 'iou': iou(mask,render_masks[v['file']])})
    for p in pairs:p.update(contour_metrics(ref_masks[p['reference_index']],render_masks[p['render']]))
    return {'camera_training_count':len(camera_training),'boundary_f1':float(np.mean([p['boundary_f1'] for p in pairs])), 'contour_distance':float(np.mean([p['contour_distance'] for p in pairs])), 'silhouette_iou': float(np.mean([p['iou'] for p in pairs])), 'pairs': pairs, 'alignment': 'fixed_from_baseline' if fixed_pairs is not None else 'shared_azimuth_search' if known else 'semantic_view_search' if labeled and candidates else 'independent_silhouette_search', 'limitations': 'Контуры нормированы по масштабу и сдвигу. Камера приблизительная; IoU не доказывает точность 3D.'}


def acceptance(before, after, margin=0.005, require_improvement=True):
    # Holdout prevents accepting a shape that only fits its generation view.
    def scores(report):
        return {p['reference_index']:p['iou'] for p in report['pairs']}
    a, b = scores(before), scores(after)
    holdout = [p['reference_index'] for p in before['pairs'] if not p.get('used_for_generation',p['reference_index']==0)]
    if not holdout:
        return False, "Нет проверочных ракурсов"
    delta = float(np.mean([b[i]-a[i] for i in holdout]))
    worst = min(b[i]-a[i] for i in holdout)
    ok = delta >= (margin if require_improvement else -.005) and worst >= -0.03
    details=[]
    for key,tolerance,direction in [('boundary_f1',.01,1),('contour_distance',.002,-1)]:
        if all(key in p for report in (before,after) for p in report['pairs']):
            first={p['reference_index']:p[key] for p in before['pairs']}
            last={p['reference_index']:p[key] for p in after['pairs']}
            change=float(np.mean([last[i]-first[i] for i in holdout]))
            if change*direction < -tolerance:ok=False
            details.append(f'{key}: {change:+.4f}')
    old_components=before.get('geometry',{}).get('components',1)
    new_components=after.get('geometry',{}).get('components',1)
    if new_components>max(4,old_components*4):
        ok=False;details.append('резко выросло число несвязных частей')
    # Thickness/flatness never chooses the direction of a semantic edit.
    return ok, ('; '.join(details)+'; ' if details else '')+f"Δ IoU проверочных видов: {delta:+.4f}; худший: {worst:+.4f}"


def validation_summary(report):
    groups={}
    for name,wanted in [('heldout',False),('used_for_generation',True)]:
        pairs=[p for p in report['pairs'] if bool(p.get('used_for_generation',p['reference_index']==0))==wanted]
        groups[name]={'count':len(pairs)}
        for key in ['iou','boundary_f1','contour_distance']:
            groups[name][key]=float(np.mean([p[key] for p in pairs])) if pairs and all(key in p for p in pairs) else None
    return groups
