"""CPU mesh edits selected by VLM, localized through visible render triangles.

Conservative candidates only: no hole filling, automatic inflation or global cuts.
All outputs still require the pipeline's before/after verification.
"""
from pathlib import Path
import json
import numpy as np
import trimesh
from PIL import Image,ImageDraw
from gai.critic_schema import EditPlan


class UnsupportedEdit(ValueError):
    pass


def raster_faces(mesh, projection, size=192):
    """Orthographic z-buffer in the exact Blender camera frame; Y increases down."""
    xyz=np.c_[mesh.vertices,np.ones(len(mesh.vertices))] @ np.asarray(projection).T
    ndc=xyz[:,:3]/xyz[:,3,None]
    xy=(ndc[:,:2]*[.5,-.5]+.5)*size-.5
    depth=np.full((size,size),np.inf);ids=np.full((size,size),-1,dtype=np.int32)
    for fid,face in enumerate(mesh.faces):
        tri=xy[face];lo=np.maximum(np.ceil(tri.min(0)).astype(int),0);hi=np.minimum(np.floor(tri.max(0)).astype(int),size-1)
        if np.any(lo>hi):continue
        a,b,c=tri;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-10:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        w=1-u-v;z=u*ndc[face[0],2]+v*ndc[face[1],2]+w*ndc[face[2],2]
        block=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1];ib=ids[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        update=(u>=-1e-7)&(v>=-1e-7)&(w>=-1e-7)&(z<block)
        block[update]=z[update];ib[update]=fid
    return ids


def selected_faces(mesh,plan,source,out):
    metrics=json.loads((source/'metrics.json').read_text())
    views={v['file']:v for v in json.loads((source/'renders/views.json').read_text())}
    pairs={p['reference_index']:p for p in metrics['pairs']}
    if len(plan.regions)<2:raise UnsupportedEdit('Нужна локализация дефекта минимум на двух ракурсах')
    support=np.zeros(len(mesh.faces),int);visible=np.zeros(len(mesh.faces),int);cameras=set()
    for region in plan.regions:
        pair=pairs.get(region.reference_index)
        if pair is None:raise UnsupportedEdit('План ссылается на неизвестное фото')
        camera=pair['render'];view=views.get(camera,{})
        if camera in cameras:raise UnsupportedEdit('Для правки нужны разные камеры, а не повтор одного рендера')
        cameras.add(camera)
        if 'projection_matrix' not in view:raise UnsupportedEdit('Нет матрицы камеры: запустите новую генерацию')
        ids=raster_faces(mesh,view['projection_matrix']);n=ids.shape[0]
        x0,y0,x1,y1=region.bbox
        mask=np.zeros_like(ids,dtype=bool);mask[int(y0*n):int(np.ceil(y1*n)),int(x0*n):int(np.ceil(x1*n))]=True
        valid=ids>=0
        total=np.bincount(ids[valid],minlength=len(mesh.faces));inside=np.bincount(ids[valid&mask],minlength=len(mesh.faces))
        visible+=total>0;support+=(total>0)&(inside/np.maximum(total,1)>=.8)
        im=Image.open(source/'renders'/camera).convert('RGB');draw=ImageDraw.Draw(im)
        draw.rectangle((x0*im.width,y0*im.height,x1*im.width,y1*im.height),outline='#e44b4b',width=3)
        im.save(out/f'region_{region.reference_index:02d}.png')
    # Require corroboration in at least two cameras, no contradictory visible view.
    selected=(support>=2)&(support==visible)
    if not selected.any():raise UnsupportedEdit('Области не подтверждают один и тот же участок сетки')
    fraction=float(mesh.area_faces[selected].sum()/mesh.area)
    if fraction>.2:raise UnsupportedEdit('Выбрано более 20% поверхности: нужна более точная локализация')
    return selected


def apply_plan(mesh_path,plan:EditPlan,source:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.load(mesh_path,force='mesh',process=False)
    mesh.visual=trimesh.visual.ColorVisuals(mesh)
    mesh.merge_vertices(merge_tex=True,merge_norm=True)
    original_vertices=mesh.vertices.copy();original_faces=mesh.faces.copy();before_faces=len(mesh.faces)
    selected=None
    if plan.operation=='repair_mesh':
        # Pure cleanup, never fill holes or discard legitimate components.
        mesh.update_faces(mesh.unique_faces() & mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices();mesh.fix_normals(multibody=True)
    elif plan.operation in {'local_scale','local_smooth','remove_region'}:
        selected=selected_faces(mesh,plan,source,out)
        (out/'selected_faces.json').write_text(json.dumps({'face_indexes':np.flatnonzero(selected).tolist()}))
        if plan.operation=='remove_region':
            mesh.update_faces(~selected);mesh.remove_unreferenced_vertices()
        else:
            # Keep region boundary fixed so a local edit cannot tear the mesh.
            touched=np.unique(mesh.faces[selected]);outside=np.unique(mesh.faces[~selected])
            interior=np.setdiff1d(touched,outside)
            if not len(interior):raise UnsupportedEdit('В области нет внутренних вершин для деформации')
            vertices=mesh.vertices.copy()
            if plan.operation=='local_scale':
                if abs(plan.scale-1)<.005:raise UnsupportedEdit('Масштаб плана не задаёт изменения')
                axes=list(range(3)) if plan.axis=='all' else ['xyz'.index(plan.axis)]
                center=vertices[touched].mean(0)
                for axis in axes:vertices[interior,axis]=center[axis]+(vertices[interior,axis]-center[axis])*plan.scale
            else:
                for i in interior:
                    neighbors=mesh.vertex_neighbors[i]
                    if len(neighbors):vertices[i]=.8*vertices[i]+.2*original_vertices[neighbors].mean(0)
            displacement=vertices-original_vertices
            limit=.03*float(np.ptp(original_vertices,axis=0).max())
            lengths=np.linalg.norm(displacement,axis=1)
            vertices=original_vertices+displacement*np.minimum(1,limit/np.maximum(lengths,1e-12))[:,None]
            mesh.vertices=vertices
    else:raise UnsupportedEdit('Операция требует повторной генерации или другого редактора')
    if len(mesh.faces)<4 or not np.isfinite(mesh.vertices).all():raise UnsupportedEdit('После правки сетка некорректна')
    if np.array_equal(mesh.faces,original_faces) and np.array_equal(mesh.vertices,original_vertices):
        raise UnsupportedEdit('Операция не изменила геометрию')
    path=out/'mesh.obj';mesh.export(path)
    (out/'edit_report.json').write_text(json.dumps({'backend':'localized CPU mesh editor','plan':plan.model_dump(),'faces_before':before_faces,'faces_after':len(mesh.faces),'selected_faces':int(selected.sum()) if selected is not None else None,'limitations':'VLM boxes are approximate; deletion can leave an open boundary. No automatic hole filling. Candidate requires independent verification.'},ensure_ascii=False,indent=2))
    return path
