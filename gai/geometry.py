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
    # Large flat caps on figurines are suspicious, but valid on machined parts.
    normals=np.asarray(mesh.face_normals)
    flat_cap=0.0
    _,axes=np.linalg.eigh(covariance)
    for axis in np.vstack((np.eye(3),axes.T)):
        projected=np.asarray(mesh.vertices)@axis
        span=max(float(np.ptp(projected)),1e-9)
        projected_tri=triangles@axis
        aligned=np.abs(normals@axis)>.97
        for bound in (projected.min(),projected.max()):
            on_plane=np.max(np.abs(projected_tri-bound),axis=1)<span*.008
            flat_cap=max(flat_cap,float(weights[on_plane & aligned].sum()))
    closed=bool(mesh.is_watertight and mesh.is_winding_consistent)
    volume=float(abs(mesh.volume)) if closed else None
    try:convex_volume=float(mesh.convex_hull.volume)
    except Exception:convex_volume=None
    return {'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),
            'euler_number':int(mesh.euler_number),'components':int(mesh.body_count),'surface_area':float(mesh.area),'volume':volume,
            'volume_to_convex_hull':volume/convex_volume if volume is not None and convex_volume and convex_volume>1e-12 else None,
            'flat_cap_fraction':flat_cap,'principal_spreads':spreads.tolist(),'thickness_ratio':ratio,'severity':severity,
            'axis_extents':np.asarray(mesh.extents).tolist(),'warning_ru':message,
            'limitations':'Эвристика толщины поверхности, не сравнение с эталонной 3D-моделью.'}


def enforce_geometry_warning(data, report):
    """Compatibility entry point: diagnostics never rewrite a VLM decision."""
    return dict(data)
