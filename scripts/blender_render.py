"""Blender 4.5 EEVEE. Canonical meshes are Y-up, centered, longest side 2.
Camera and scale are fixed for all iterations; no recentering after a deformation.
"""
import sys
import json
import math
from pathlib import Path
import bpy
from mathutils import Vector


def main():
    job=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.render.engine='BLENDER_EEVEE_NEXT'
    scene.render.resolution_x=scene.render.resolution_y=job['resolution']
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGBA'
    scene.render.film_transparent=True
    scene.view_settings.view_transform='AgX'
    scene.world=bpy.data.worlds.new('World');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(0.3,0.3,0.3,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0.6
    verts=[];faces=[]
    for line in Path(job['mesh']).read_text().splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='v':
            x,y,z=map(float,fields[1:4]);verts.append((x,-z,y))
        elif fields[0]=='f':
            face=[]
            for field in fields[1:]:
                idx=int(field.split('/')[0]);face.append(idx-1 if idx>0 else len(verts)+idx)
            faces.append(face)
    data=bpy.data.meshes.new('Geometry');data.from_pydata(verts,[],faces);data.update()
    obj=bpy.data.objects.new('Object',data);scene.collection.objects.link(obj)
    mat=bpy.data.materials.new('Clay');mat.diffuse_color=(0.3,0.38,0.44,1);mat.use_nodes=True
    shader=mat.node_tree.nodes.get('Principled BSDF');shader.inputs['Base Color'].default_value=(0.3,0.38,0.44,1);shader.inputs['Roughness'].default_value=0.85
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:polygon.use_smooth=True
    for name,loc,power,size in [('Key',(4,-4,6),700,5),('Fill',(-4,-2,3),400,4),('Rim',(0,5,4),500,3)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size
        source=bpy.data.objects.new(name,light);scene.collection.objects.link(source);source.location=loc;source.rotation_euler=(-source.location).to_track_quat('-Z','Y').to_euler()
    camera=bpy.data.cameras.new('Camera');cam=bpy.data.objects.new('Camera',camera);scene.collection.objects.link(cam);scene.camera=cam
    camera.type='ORTHO';camera.ortho_scale=3.1;camera.lens=50
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True);views=[]
    for elev in sorted(set(job['elevations']) | {0,90}):
        for az in ([0] if elev==90 else sorted(set(range(0,360,job['step'])) | {0,90,180,270})):
            a=math.radians(az);e=math.radians(elev)
            cam.location=Vector((5*math.sin(a)*math.cos(e),-5*math.cos(a)*math.cos(e),5*math.sin(e)))
            cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler()
            name=f'az_{az:03d}_el_{elev:02d}.png';scene.render.filepath=str(out/name)
            bpy.ops.render.render(write_still=True)
            views.append({'file':name,'azimuth_deg':az,'elevation_deg':elev,'camera_location':list(cam.location),'orthographic_scale':camera.ortho_scale})
    (out/'views.json').write_text(json.dumps(views,indent=2))


if __name__=='__main__':main()
