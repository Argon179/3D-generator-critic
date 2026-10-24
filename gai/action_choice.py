"""Finite feasible actions: the language model selects a token, code compiles a plan."""
LABELS={'M':'neural_refine','A':'fit_silhouette','B':'local_smooth','R':'regenerate','L':'local_review','N':'none','C':'camera_mismatch'}


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


def select_report(choices,global_report=None):
    """Respect the whole-object routing decision, including refusal and pose veto."""
    if global_report is not None and global_report['confidence']>=.65:return global_report
    active=[c for c in choices if c['should_edit']]
    return max(active or choices,key=lambda c:c['confidence'])


def decision_data(label,probabilities,proposal_id,permitted,reason_ru=None):
    allowed=action_labels(permitted)
    if label not in allowed:raise ValueError('VLM selected an unavailable action')
    action=allowed[label];camera=action=='camera_mismatch'
    if action=='local_review':raise ValueError('Local review is a routing choice, not an executable edit')
    confidence=float(probabilities[label])
    if not 0<=confidence<=1:raise ValueError('Invalid action probability')
    if not reason_ru:
        reason_ru={'fit_silhouette':'VLM chose local contour fitting using the selected views.',
                   'neural_refine':'VLM chose photo-anchored diffusion deformation of the existing mesh.',
                   'local_smooth':'VLM chose local surface smoothing of the selected region.',
                   'regenerate':'VLM chose reconstruction from the original photos because local editing is insufficient.',
                   'none':'VLM found no useful local operation for this region.',
                   'camera_mismatch':'VLM indicated a camera mismatch; geometry remains unchanged.'}[action]
    return {'action':'none' if camera else action,'proposal_id':proposal_id if action not in {'none','camera_mismatch'} else None,
            'confidence':confidence,'camera_mismatch':camera,'reason_ru':reason_ru}
