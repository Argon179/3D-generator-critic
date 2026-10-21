"""Coarse silhouette alignment. Never treat this as 3D ground truth."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image


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


def align(references, render_dir, fixed_pairs=None):
    render_dir = Path(render_dir)
    views = json.loads((render_dir / "views.json").read_text())
    render_masks = {v['file']: normalized_mask(render_dir / v['file']) for v in views}
    ref_masks = [normalized_mask(r['prepared']) for r in references]
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
                        score = np.mean([p['iou'] for p in candidate]) if len(candidate) == len(references) else -1
                        if score > best:
                            best, pairs = score, candidate
        if pairs is None:
            # Unknown camera: independent nearest silhouette, explicitly low confidence.
            pairs = []
            for k, mask in enumerate(ref_masks):
                v = max(views, key=lambda v:iou(mask,render_masks[v['file']]))
                pairs.append({'reference_index': k, 'render': v['file'], 'azimuth_deg': v['azimuth_deg'], 'elevation_deg': v['elevation_deg'], 'iou': iou(mask,render_masks[v['file']])})
    return {'silhouette_iou': float(np.mean([p['iou'] for p in pairs])), 'pairs': pairs, 'alignment': 'fixed_from_baseline' if fixed_pairs is not None else 'shared_azimuth_search' if known else 'independent_silhouette_search', 'limitations': 'Контуры нормированы по масштабу и сдвигу. Камера приблизительная; IoU не доказывает точность 3D.'}


def acceptance(before, after, margin=0.005):
    # Holdout prevents accepting a shape that only fits its generation view.
    def scores(report):
        return {p['reference_index']:p['iou'] for p in report['pairs']}
    a, b = scores(before), scores(after)
    holdout = [i for i in a if i != 0]
    if not holdout:
        return False, "Нет проверочных ракурсов"
    delta = float(np.mean([b[i]-a[i] for i in holdout]))
    worst = min(b[i]-a[i] for i in holdout)
    ok = delta >= margin and worst >= -0.03
    return ok, f"Δ IoU проверочных видов: {delta:+.4f}; худший: {worst:+.4f}"
