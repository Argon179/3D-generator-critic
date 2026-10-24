"""Camera-space evidence and executable proposal IDs, independent of object names."""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
import trimesh
from gai.local_editor import raster_faces


def foreground(path):
    im=Image.open(path).convert('RGBA');a=np.asarray(im)
    if a[:,:,3].min()==a[:,:,3].max():raise ValueError('Для сопоставления формы нужна альфа-маска')
    return im,a[:,:,3]>127


def aligned_reference(reference,render,size=192):
    """Freeze similarity transform from the BASELINE render, never refit a candidate."""
    ref,rm=foreground(reference);rend,bm=foreground(render)
    ry,rx=np.nonzero(rm);by,bx=np.nonzero(bm)
    if not len(rx) or not len(bx):raise ValueError('Пустой силуэт')
    rbox=(rx.min(),ry.min(),rx.max()+1,ry.max()+1)
    bbox=(bx.min(),by.min(),bx.max()+1,by.max()+1)
    scale=max(bbox[2]-bbox[0],bbox[3]-bbox[1])/max(rbox[2]-rbox[0],rbox[3]-rbox[1])
    crop=ref.crop(rbox);crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS)
    canvas=Image.new('RGBA',rend.size)
    center=((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)
    canvas.alpha_composite(crop,(round(center[0]-crop.width/2),round(center[1]-crop.height/2)))
    return canvas.resize((size,size),Image.Resampling.LANCZOS)


def project(vertices,matrix):
    q=np.c_[vertices,np.ones(len(vertices))] @ np.asarray(matrix).T
    ndc=q[:,:3]/q[:,3,None]
    return ndc[:,:2]*[.5,-.5]+.5


def region_vertices(mesh,views,regions):
    """Intersect viewing volumes, not front/back surface IDs (which never coincide)."""
    inside=[];visible=np.zeros(len(mesh.vertices),bool)
    for region in regions:
        view=views[region['reference_index']]
        uv=project(mesh.vertices,view['projection_matrix']);x0,y0,x1,y1=region['bbox']
        inside.append((uv[:,0]>=x0)&(uv[:,0]<=x1)&(uv[:,1]>=y0)&(uv[:,1]<=y1))
        ids=view['face_ids'];fid=np.unique(ids[ids>=0]);visible[np.unique(mesh.faces[fid])]=True
    return np.logical_and.reduce(inside)&visible


def build_catalog(mesh_path,folder,pairs):
    folder=Path(folder);out=folder/'shape_evidence';out.mkdir(exist_ok=True)
    mesh=trimesh.load(mesh_path,force='mesh',process=False)
    view_info={v['file']:v for v in json.loads((folder/'renders/views.json').read_text())}
    views={};patches=[];records=[]
    for pair in pairs:
        idx=pair['reference_index'];v=view_info[pair['render']]
        if 'projection_matrix' not in v:continue
        baseline=folder.parent/'iteration_00/shape_evidence'/f'target_{idx:02d}.png'
        target=Image.open(baseline).convert('RGBA') if folder.name!='iteration_00' and baseline.is_file() else aligned_reference(pair['reference_path'],folder/'renders'/pair['render'])
        target.save(out/f'target_{idx:02d}.png')
        _,current=foreground(folder/'renders'/pair['render'])
        current=np.asarray(Image.fromarray(current).resize((192,192),Image.Resampling.NEAREST),bool)
        reference=np.asarray(target)[:,:,3]>127
        views[idx]=dict(v,face_ids=raster_faces(mesh,v['projection_matrix']))
        # Suppress tiny anti-aliasing residuals; retain genuine protrusions/indentations.
        raw=current^reference;change=ndimage.binary_opening(raw,iterations=1)
        groups,n=ndimage.label(ndimage.binary_dilation(change,iterations=3))
        overlay=np.full((192,192,3),235,dtype=np.uint8)
        overlay[current|reference]=[150,165,165]
        overlay[current&~reference]=[235,65,65]
        overlay[reference&~current]=[40,160,220]
        Image.fromarray(overlay).save(out/f'difference_{idx:02d}.png')
        record={'reference_index':idx,'render':pair['render'],'target':str(out/f'target_{idx:02d}.png'),
                'difference':str(out/f'difference_{idx:02d}.png'),'used_for_generation':pair.get('used_for_generation',idx==0),'projection_matrix':v['projection_matrix']}
        records.append(record)
        components=[]
        for k in range(1,n+1):
            area=int(np.count_nonzero(change&(groups==k)))
            if area<12:continue
            yy,xx=np.nonzero(groups==k)
            box=[max(0,(xx.min()-6)/192),max(0,(yy.min()-6)/192),min(1,(xx.max()+7)/192),min(1,(yy.max()+7)/192)]
            components.append({'reference_index':idx,'bbox':box,'pixels':area})
        patches.extend(sorted(components,key=lambda x:-x['pixels'])[:4])
    controls=[p['reference_index'] for p in pairs if not p.get('used_for_generation',p['reference_index']==0)]
    reserved_control=controls[len(controls)//2] if controls else None
    proposals=[]
    for i,a in enumerate(patches):
        for b in patches[i+1:]:
            ia,ib=a['reference_index'],b['reference_index']
            if ia==ib or views[ia]['file']==views[ib]['file'] or reserved_control in {ia,ib}:continue
            # Avoid nearly coincident views pretending to be independent evidence.
            pa=np.asarray(views[ia]['camera_location']);pb=np.asarray(views[ib]['camera_location'])
            angle=np.degrees(np.arccos(np.clip(np.dot(pa,pb)/np.linalg.norm(pa)/np.linalg.norm(pb),-1,1)))
            if angle<25:continue
            regions=[{k:r[k] for k in ['reference_index','bbox']} for r in [a,b]]
            selected=region_vertices(mesh,views,regions)
            count=int(selected.sum());fraction=count/len(mesh.vertices)
            if count<8 or fraction>.35:continue
            # Two patch volumes must overlap actual vertices near the affected outlines.
            score=(a['pixels']+b['pixels'])/(1+fraction)
            proposals.append({'regions':regions,'vertex_indexes':np.flatnonzero(selected).tolist(),'score':score,'views_angle_deg':float(angle)})
    proposals.sort(key=lambda p:-p['score'])
    chosen=[]
    for p in proposals:
        ids=set(p['vertex_indexes'])
        if any(len(ids&set(q['vertex_indexes']))/max(1,len(ids|set(q['vertex_indexes'])))>.65 for q in chosen):continue
        p['id']=f'contour_{len(chosen)+1:02d}'
        p['operation']='fit_silhouette';p['evidence']='paired silhouette residuals, not color or texture'
        montage=Image.new('RGB',(768,384),'#eee');draw=ImageDraw.Draw(montage)
        for row,r in enumerate(p['regions']):
            idx=r['reference_index'];rec=next(x for x in records if x['reference_index']==idx)
            for col,path in enumerate([rec['target'],str(folder/'renders'/rec['render']),rec['difference']]):
                im=Image.open(path).convert('RGBA');bg=Image.new('RGBA',im.size,'#eee');bg.alpha_composite(im)
                im=bg.convert('RGB').resize((256,256))
                # 384-high montage has two 192px rows, three 256px columns.
                im=im.resize((256,192));montage.paste(im,(col*256,row*192))
            x0,y0,x1,y1=r['bbox']
            for col in range(3):draw.rectangle((col*256+x0*256,row*192+y0*192,col*256+x1*256,row*192+y1*192),outline='#252525',width=2)
            draw.text((3,row*192+3),f'view {idx}: reference | mesh | excess red / missing blue',fill='black')
        image_path=out/(p['id']+'.png');montage.save(image_path);p['image']=str(image_path)
        p['focus_images']=[]
        for region in p['regions']:
            idx=region['reference_index'];rec=next(x for x in records if x['reference_index']==idx)
            x0,y0,x1,y1=region['bbox'];cx=(x0+x1)/2;cy=(y0+y1)/2;radius=max(x1-x0,y1-y0)*.65
            crop=(max(0,cx-radius),max(0,cy-radius),min(1,cx+radius),min(1,cy+radius))
            for kind,path in [('reference',rec['target']),('mesh',str(folder/'renders'/rec['render'])),('difference',rec['difference'])]:
                im=Image.open(path).convert('RGBA');bg=Image.new('RGBA',im.size,'#eee');bg.alpha_composite(im)
                box=tuple(round(v*(im.width if i%2==0 else im.height)) for i,v in enumerate(crop))
                im=bg.convert('RGB').crop(box);im.thumbnail((384,384));im=im.resize((max(1,im.width*384//max(im.size)),max(1,im.height*384//max(im.size))))
                path=out/f'{p["id"]}_view_{idx}_{kind}.png';im.save(path);p['focus_images'].append(str(path))
        chosen.append(p)
        if len(chosen)==4:break
    catalog={'version':1,'reserved_control':reserved_control,'mesh_sha256':hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest(),'source_mesh':str(mesh_path),'views':records,'proposals':chosen,
             'limitations':'Camera and reference similarity transform are approximate; VLM must distinguish pose mismatch from shape mismatch.'}
    (out/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
    return catalog
