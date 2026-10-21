"""Input planning shared by the API and isolated multiview workers."""
from __future__ import annotations
import math


def select_views(refs, backend, explicit=None):
    if backend == 'hunyuan':
        if explicit:
            if set(explicit) - {'front', 'left', 'back', 'right'}:
                raise ValueError('Допустимые стороны: front, left, back, right')
            chosen = {k: int(v) for k, v in explicit.items()}
            if 'front' not in chosen or len(chosen) < 2:
                raise ValueError('Укажите передний вид и минимум ещё одну сторону')
            if len(set(chosen.values())) != len(chosen) or any(i < 0 or i >= len(refs) for i in chosen.values()):
                raise ValueError('Каждая сторона должна ссылаться на отдельное выбранное фото')
        else:
            if len(refs) < 5 or any(r.get('azimuth_deg') is None for r in refs):
                raise ValueError('Назначьте стороны под фотографиями: минимум «Спереди» и ещё одну сторону, плюс отдельное фото «Для проверки». Для автоподбора выберите минимум 5 фото с известными углами.')
            # Front is the user-selected main photo. Angles increase around Y.
            origin = float(refs[0]['azimuth_deg'])
            chosen = {}
            for side, angle in [('front', 0), ('left', 90), ('back', 180), ('right', 270)]:
                distances = [abs((float(r['azimuth_deg']) - origin - angle + 180) % 360 - 180) for r in refs]
                index = min(range(len(refs)), key=distances.__getitem__)
                if distances[index] > 20:
                    raise ValueError(f'Нет вида {side} в пределах 20°; задайте стороны вручную')
                chosen[side] = index
        if len(chosen) >= len(refs):
            raise ValueError('Оставьте минимум одно дополнительное фото для независимой проверки')
        return chosen
    if backend == 'da3':
        if len(refs) < 3:
            raise ValueError('Для DA3 нужны минимум 3 фото: 2 входных и 1 проверочное')
        # Evenly reserve roughly a quarter, excluding the main image.
        count = max(1, len(refs) // 4)
        heldout = {1 + math.floor(i * (len(refs) - 2) / max(1, count - 1)) for i in range(count)}
        return [i for i in range(len(refs)) if i not in heldout]
    raise ValueError('Неизвестный multiview-генератор')


def cached_snapshot(cache, model, files, revision=None):
    """Resolve complete local weights; inference never silently downloads new weights."""
    from pathlib import Path
    base=Path(cache)/'hub'/('models--'+model.replace('/','--'))
    candidates=[]
    if revision:
        candidates=[base/'snapshots'/revision]
    else:
        main=base/'refs/main'
        if main.is_file():candidates.append(base/'snapshots'/main.read_text().strip())
        candidates.extend(sorted((base/'snapshots').glob('*')))
    return next((p for p in candidates if all((p/f).is_file() for f in files)),None)
