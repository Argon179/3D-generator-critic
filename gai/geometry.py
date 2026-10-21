"""Rotation-invariant shape diagnostics, not a measure of reconstruction accuracy."""
import numpy as np


def geometry_report(mesh):
    triangles=np.asarray(mesh.triangles,dtype=np.float64)
    weights=np.asarray(mesh.area_faces,dtype=np.float64)
    if not len(weights) or weights.sum()<=1e-12:
        raise ValueError('Нет поверхности для проверки объёма')
    weights=weights/weights.sum()
    # Exact surface moments of triangles: independent of vertex density/UV seams.
    center=np.einsum('i,ij->j',weights,triangles.mean(axis=1))
    t=triangles-center
    summed=t.sum(axis=1)
    second=(np.einsum('nki,nkj->nij',t,t)+np.einsum('ni,nj->nij',summed,summed))/12
    covariance=np.einsum('n,nij->ij',weights,second)
    eigenvalues=np.maximum(np.linalg.eigvalsh(covariance),0)
    spreads=np.sqrt(eigenvalues)
    ratio=float(spreads[0]/max(spreads[-1],1e-12))
    severity='severe' if ratio<.12 else 'suspect' if ratio<.35 else 'none'
    message=(f'Проверка геометрии: малая толщина, отношение разбросов по главным осям {ratio:.3f}. '
             'Возможна плоская реконструкция; для объёмной фигурки это дефект. '
             'Для намеренно тонкой детали такая форма может быть правильной. '
             'Нужна проверка бокового и верхнего видов.' if severity!='none' else
             'Выраженная плоскостность не обнаружена; это не подтверждает правильность формы.')
    return {'principal_spreads':spreads.tolist(),'thickness_ratio':ratio,'severity':severity,
            'axis_extents':np.asarray(mesh.extents).tolist(),'warning_ru':message,
            'limitations':'Эвристика толщины поверхности, не сравнение с эталонной 3D-моделью.'}


def enforce_geometry_warning(data, report):
    data=dict(data)
    if report['severity']=='none':
        return data
    data['summary_ru']=(report['warning_ru']+' Оценка VLM: '+data['summary_ru'])[:2000]
    data['issues']=[{'part':'Объём / толщина','observation_ru':report['warning_ru'],
                    'suggestion_ru':'Проверьте независимые виды; при сплющенной фигурке повторите генерацию по другому ракурсу.'},*data.get('issues',[])][:10]
    # A generic CLIP deformation is not a reliable way to reconstruct a missing side.
    data['should_edit']=False
    data['target_description']=data['base_description']
    return data
