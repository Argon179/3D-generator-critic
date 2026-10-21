"""Approximate orthographic visual hull from known turntable azimuths.

No learned detail or texture; concavities invisible in silhouettes cannot be recovered.
"""
import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh


def normalized_silhouette(path,size=192):
    a=np.asarray(Image.open(path).convert('RGBA'))[:,:,3]>127
    y,x=np.nonzero(a)
    if not len(x):raise ValueError('Пустой силуэт')
    # Object height provides a shared scale across views; never scale by width.
    crop=Image.fromarray(a[y.min():y.max()+1,x.min():x.max()+1])
    factor=(size*.78)/crop.height
    crop=crop.resize((max(1,round(crop.width*factor)),round(crop.height*factor)),Image.Resampling.NEAREST)
    if crop.width>size:raise ValueError('Слишком широкий объект для калибровки по высоте')
    canvas=Image.new('L',(size,size));canvas.paste(crop,((size-crop.width)//2,(size-crop.height)//2))
    return np.asarray(canvas)>0


def visual_hull(references,resolution=112,elevation=0):
    from skimage.measure import marching_cubes
    if len(references)<4 or any(r.get('azimuth_deg') is None for r in references):
        raise ValueError('Нужны минимум четыре известных азимута')
    angles=np.sort(np.unique([r['azimuth_deg']%360 for r in references]))
    if np.diff(np.r_[angles,angles[0]+360]).max()>180:
        raise ValueError('Недостаточное покрытие вокруг объекта')
    extent=3.2;axis=np.linspace(-extent/2,extent/2,resolution,dtype=np.float32)
    x,y,z=np.meshgrid(axis,axis,axis,indexing='ij')
    field=np.full(x.shape,1000,dtype=np.float32)
    e=np.radians(elevation)
    for ref in references:
        mask=normalized_silhouette(ref['prepared'])
        # Small mask tolerance for segmentation and approximate orthographic pose.
        mask=ndimage.binary_dilation(mask,iterations=1)
        a=np.radians(ref['azimuth_deg'])
        horizontal=x*np.cos(a)-z*np.sin(a)
        vertical=y*np.cos(e)-(x*np.sin(a)+z*np.cos(a))*np.sin(e)
        u=(mask.shape[1]-1)/2+horizontal*(mask.shape[0]*.78/2)
        v=(mask.shape[0]-1)/2-vertical*(mask.shape[0]*.78/2)
        signed=ndimage.distance_transform_edt(mask)-ndimage.distance_transform_edt(~mask)
        distances=ndimage.map_coordinates(signed,[v.ravel(),u.ravel()],order=1,mode='constant',cval=-100).reshape(x.shape)
        field=np.minimum(field,distances)
    occupancy=field>0
    labels,count=ndimage.label(occupancy)
    sizes=np.bincount(labels.ravel());sizes[0]=0
    if not count or sizes.max()<100:raise ValueError('Несогласованные маски: объём не восстановлен')
    field[labels!=sizes.argmax()]=np.minimum(field[labels!=sizes.argmax()],-1)
    field=ndimage.gaussian_filter(np.pad(field,1,constant_values=-100),sigma=1.0)
    verts,faces,_,_=marching_cubes(field,level=0,spacing=(extent/(resolution-1),)*3)
    verts-=extent/2+extent/(resolution-1)
    mesh=trimesh.Trimesh(verts,faces,process=True)
    trimesh.smoothing.filter_taubin(mesh,iterations=10)
    mesh.fix_normals()
    return mesh
