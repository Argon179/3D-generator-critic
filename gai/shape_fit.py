"""Bounded multi-view contour fitting of an existing mesh (CPU sparse least squares).
No semantic body/axis assumptions, no new topology, only VLM-selected proposal vertices.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy import sparse,ndimage
from scipy.sparse.linalg import lsqr
import trimesh
from gai.local_editor import raster_faces,UnsupportedEdit
from gai.shape_evidence import project,foreground


def mask_score(mesh,records):
    scores={}
    for r in records:
        predicted=raster_faces(mesh,r['projection_matrix'])>=0
        _,target=foreground(r['target'])
        scores[r['reference_index']]=float((predicted&target).sum()/max(1,(predicted|target).sum()))
    return scores


def fit_mesh(mesh_path,source,out,proposal_id,steps=14):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((source/'shape_evidence/catalog.json').read_text())
    if catalog.get('mesh_sha256')!=hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest():
        raise UnsupportedEdit('Области относятся к другой версии сетки')
    proposal=next((p for p in catalog['proposals'] if p['id']==proposal_id),None)
    if proposal is None:raise UnsupportedEdit('VLM выбрала неизвестный вариант правки')
    mesh=trimesh.load(mesh_path,force='mesh',process=False)
    selected=np.asarray(proposal['vertex_indexes'],int);original=mesh.vertices.copy()
    local=np.full(len(original),-1,int);local[selected]=np.arange(len(selected))
    records={r['reference_index']:r for r in catalog['views']}
    training=[records[r['reference_index']] for r in proposal['regions']]
    if catalog.get('reserved_control') in {r['reference_index'] for r in training}:raise UnsupportedEdit('Этот ракурс зарезервирован только для контроля')
    if len(training)<2 or len(training)>=len(records):raise UnsupportedEdit('Нужны два обучающих и хотя бы один контрольный ракурс')
    if len(selected)<8 or len(selected)>.35*len(original):raise UnsupportedEdit('Неподходящий размер локальной области')
    before=mask_score(mesh,catalog['views']);history=[]
    reliable=(mesh.area_faces>max(1e-14,mesh.area*1e-12)) & (np.linalg.norm(mesh.face_normals,axis=1)>.9)
    edges={}
    for r in training:
        _,target=foreground(r['target'])
        signed=ndimage.distance_transform_edt(~target)-ndimage.distance_transform_edt(target)
        signed=ndimage.gaussian_filter(signed.astype(float),1)
        gy,gx=np.gradient(signed)
        edges[r['reference_index']]=(signed,gx,gy)
    best_vertices=original.copy();best_score=np.mean([before[r['reference_index']] for r in training])
    span=float(np.ptp(original,axis=0).max());max_step=.015*span;max_total=.1*span
    distance=np.full(len(original),-1,int);outside=local<0;distance[outside]=0
    frontier=np.flatnonzero(outside).tolist()
    for level in range(1,5):
        following=[]
        for vi in frontier:
            for j in mesh.vertex_neighbors[vi]:
                if distance[j]<0:distance[j]=level;following.append(j)
        frontier=following
    feather=np.where(distance[selected]<0,1,np.minimum(1,distance[selected]/4))
    # Regularize displacements (not vertex positions), preserving the original shape.
    n=len(selected);rr=[];cc=[];dd=[];row=0
    edges_all=mesh.edges_unique
    for va,vb in edges_all:
        if local[va]<0 and local[vb]<0:continue
        length=np.linalg.norm(original[va]-original[vb])
        weight=.025/max(length,.001*span)
        for axis in range(3):
            if local[va]>=0:rr.append(row);cc.append(local[va]*3+axis);dd.append(weight)
            if local[vb]>=0:rr.append(row);cc.append(local[vb]*3+axis);dd.append(-weight)
            row+=1
    lap=sparse.coo_matrix((dd,(rr,cc)),shape=(row,n*3)).tocsr()
    anchor=sparse.eye(n*3,format='csr')*.025
    for iteration in range(steps):
        rows=[];cols=[];values=[];rhs=[];row=0
        for r in training:
            matrix=np.asarray(r['projection_matrix']);ids=raster_faces(mesh,matrix);mask=ids>=0
            filled=ndimage.binary_fill_holes(mask)
            edge=filled & ~ndimage.binary_erosion(filled)
            edge &= mask
            yy,xx=np.nonzero(edge);faces=ids[yy,xx]
            eligible=(local[mesh.faces[faces]]>=0).any(axis=1)
            yy,xx,faces=yy[eligible],xx[eligible],faces[eligible]
            if len(yy)>600:
                take=np.linspace(0,len(yy)-1,600,dtype=int);yy,xx,faces=yy[take],xx[take],faces[take]
            uv=project(mesh.vertices,matrix)*192-.5
            signed,gx,gy=edges[r['reference_index']]
            P=np.array([matrix[0,:3]*.5,-matrix[1,:3]*.5])
            for y,x,fi in zip(yy,xx,faces):
                face=mesh.faces[fi];tri=uv[face]
                system=np.vstack([tri.T,np.ones(3)])
                try:bary=np.linalg.solve(system,np.array([x,y,1]))
                except np.linalg.LinAlgError:continue
                if np.min(bary)<-.01:continue
                normal=np.array([gx[y,x],gy[y,x]])
                length=np.linalg.norm(normal)
                if length<1e-5 or abs(signed[y,x])<.5:continue
                normal/=length
                # Point-to-plane distance avoids tangential pulling and sharp spikes
                # from chasing unrelated corners of the reference outline.
                projected=normal@P
                for vi,b in zip(face,bary):
                    if local[vi]<0:continue
                    for xyz in range(3):
                        rows.append(row);cols.append(local[vi]*3+xyz);values.append(float(b*projected[xyz]))
                rhs.append(float(-signed[y,x]/192));row+=1
        if row<4:
            history.append({'step':iteration+1,'reason':'no boundary constraints','constraints':row});break
        A=sparse.coo_matrix((values,(rows,cols)),shape=(row,n*3)).tocsr()
        current=mesh.vertices[selected]-original[selected]
        # Smooth cumulative displacement, keeping all unselected vertices fixed.
        system=sparse.vstack([A,lap,anchor],format='csr')
        target=np.r_[rhs,-lap@current.ravel(),-anchor@current.ravel()]
        delta=lsqr(system,target,atol=1e-5,btol=1e-5,iter_lim=120)[0].reshape(-1,3)
        lengths=np.linalg.norm(delta,axis=1);delta*=np.minimum(1,max_step/np.maximum(lengths,1e-12))[:,None]
        previous=mesh.vertices.copy();old_normals=mesh.face_normals.copy()
        valid=False
        for factor in [1,.5,.25,.125,.0625,.03125,.015625,.0078125]:
            vertices=previous.copy();vertices[selected]+=delta*factor
            displacement=vertices[selected]-original[selected]
            lengths=np.linalg.norm(displacement,axis=1)
            vertices[selected]=original[selected]+displacement*np.minimum(1,max_total/np.maximum(lengths,1e-12))[:,None]
            vertices[selected]=previous[selected]+(vertices[selected]-previous[selected])*feather[:,None]
            mesh.vertices=vertices
            if np.all(mesh.area_faces[reliable]>1e-14) and np.min(np.sum(mesh.face_normals[reliable]*old_normals[reliable],axis=1))>.1:
                valid=True;break
        if not valid:
            mesh.vertices=previous;history.append({'step':iteration+1,'reason':'line search rejected'});break
        scores=mask_score(mesh,training);score=float(np.mean(list(scores.values())))
        history.append({'step':iteration+1,'training_iou':score})
        if score>best_score+1e-5:best_score=score;best_vertices=mesh.vertices.copy()
    mesh.vertices=best_vertices
    displacement=np.linalg.norm(best_vertices-original,axis=1)
    if displacement.max()<1e-7:
        (out/'fit_diagnostics.json').write_text(json.dumps({'before':before,'history':history},indent=2))
        raise UnsupportedEdit('Локальная подгонка не улучшила обучающие контуры')
    result=out/'mesh.obj';mesh.export(result)
    after=mask_score(mesh,catalog['views'])
    report={'backend':'multi-view sparse contour fitting','plan':{'operation':'fit_silhouette','proposal_id':proposal_id,'regions':proposal['regions']},
            'training_views':[r['reference_index'] for r in training],'control_views':[i for i in records if i not in {r['reference_index'] for r in training} and not records[i].get('used_for_generation',False)],
            'unoptimized_views':[i for i in records if i not in {r['reference_index'] for r in training}],
            'before_fixed_frame_iou':before,'after_fixed_frame_iou':after,'max_displacement':float(displacement.max()),
            'reserved_control':catalog.get('reserved_control'),'selected_vertices':len(selected),'topology_preserved':True,'history':history,
            'limitations':'Fits observed outlines using approximate cameras. Does not infer hidden surfaces or exact CAD dimensions.'}
    (out/'edit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return result


def smooth_mesh(mesh_path,source,out,proposal_id,iterations=16):
    """Taubin smoothing only in the VLM-selected patch, with a fixed boundary."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((Path(source)/'shape_evidence/catalog.json').read_text())
    if catalog.get('mesh_sha256')!=hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest():raise UnsupportedEdit('Область относится к другой сетке')
    proposal=next((p for p in catalog['proposals'] if p['id']==proposal_id),None)
    if proposal is None:raise UnsupportedEdit('Неизвестный ID области')
    mesh=trimesh.load(mesh_path,force='mesh',process=False);original=mesh.vertices.copy()
    selected=np.zeros(len(original),bool);selected[proposal['vertex_indexes']]=True
    if selected.mean()>.35:raise UnsupportedEdit('Слишком большая область сглаживания')
    # A fixed boundary prevents abrupt seams and preserves adjacent features.
    active=selected.copy()
    for vi in np.flatnonzero(selected):
        if not selected[mesh.vertex_neighbors[vi]].all():active[vi]=False
    rr=[];cc=[];vv=[]
    for i,neighbors in enumerate(mesh.vertex_neighbors):
        for j in neighbors:rr.append(i);cc.append(j);vv.append(1/len(neighbors))
    operator=sparse.coo_matrix((vv,(rr,cc)),shape=(len(original),len(original))).tocsr()
    before=mask_score(mesh,catalog['views']);reliable=(mesh.area_faces>max(1e-14,mesh.area*1e-12))&(np.linalg.norm(mesh.face_normals,axis=1)>.9)
    max_total=.035*np.ptp(original,axis=0).max()
    for _ in range(iterations):
        old=mesh.vertices.copy();normals=mesh.face_normals.copy();vertices=old.copy()
        for strength in [.45,-.47]:
            update=operator@vertices-vertices
            vertices[active]+=strength*update[active]
        delta=vertices-original;length=np.linalg.norm(delta,axis=1)
        vertices=original+delta*np.minimum(1,max_total/np.maximum(length,1e-12))[:,None]
        valid=False
        for fraction in [1,.5,.25,.125]:
            mesh.vertices=old+(vertices-old)*fraction
            if np.all(mesh.area_faces[reliable]>1e-14) and np.min(np.sum(mesh.face_normals[reliable]*normals[reliable],axis=1))>.1:
                valid=True;break
        if not valid:mesh.vertices=old;break
    displacement=np.linalg.norm(mesh.vertices-original,axis=1)
    if displacement.max()<1e-6:raise UnsupportedEdit('Сглаживание не изменило область')
    path=out/'mesh.obj';mesh.export(path)
    report={'backend':'localized Taubin smoothing','plan':{'operation':'local_smooth','proposal_id':proposal_id,'regions':proposal['regions']},
            'training_views':[],'control_views':[r['reference_index'] for r in catalog['views'] if not r.get('used_for_generation',False)],
            'unoptimized_views':[r['reference_index'] for r in catalog['views']],
            'before_fixed_frame_iou':before,'after_fixed_frame_iou':mask_score(mesh,catalog['views']),
            'max_displacement':float(displacement.max()),'selected_vertices':int(active.sum()),'topology_preserved':True,
            'limitations':'Local surface smoothing; preserves topology and outside vertices. Does not add missing parts.'}
    (out/'edit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return path
