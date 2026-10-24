"""Finite feasible actions: the language model selects a token, code compiles a plan."""
LABELS={'A':'fit_silhouette','B':'local_smooth','R':'regenerate','N':'none','C':'camera_mismatch'}


def global_pairs(pairs,limit=3):
    """Spread full-view evidence around the object instead of adjacent photos."""
    if not pairs:return []
    selected=[pairs[0]];remaining=pairs[1:]
    while remaining and len(selected)<limit:
        def separation(pair):
            angle=pair.get('camera',{}).get('azimuth_deg')
            if angle is None:return -1
            known=[p.get('camera',{}).get('azimuth_deg') for p in selected]
            return min((abs((angle-a+180)%360-180) for a in known if a is not None),default=-1)
        chosen=max(remaining,key=separation);selected.append(chosen);remaining=[p for p in remaining if p is not chosen]
    return selected


def action_labels(permitted):
    return {k:v for k,v in LABELS.items() if v in permitted or k in {'N','C'}}


def decision_data(label,probabilities,proposal_id,permitted,reason_ru=None):
    allowed=action_labels(permitted)
    if label not in allowed:raise ValueError('VLM selected an unavailable action')
    action=allowed[label];camera=action=='camera_mismatch'
    confidence=float(probabilities[label])
    if not 0<=confidence<=1:raise ValueError('Invalid action probability')
    if not reason_ru:
        reason_ru={'fit_silhouette':'VLM выбрала локальную подгонку контура по выбранным ракурсам.',
                   'local_smooth':'VLM выбрала локальное сглаживание поверхности выбранного участка.',
                   'regenerate':'VLM выбрала повторную реконструкцию по исходным фотографиям, поскольку локальная правка недостаточна.',
                   'none':'VLM не выбрала полезную локальную операцию для этой области.',
                   'camera_mismatch':'VLM указала на несогласованный ракурс; геометрия не меняется.'}[action]
    return {'action':'none' if camera else action,'proposal_id':proposal_id if action not in {'none','camera_mismatch'} else None,
            'confidence':confidence,'camera_mismatch':camera,'reason_ru':reason_ru}
