import blenderproc as bproc
import bpy
import bmesh

import numpy as np
import os
from math import radians
from mathutils import Vector
# BlenderProc 초기화
bproc.init()




root_dir = "/nas/ochansol/3d_model/2025_aw_box/org"
dir_list = [i for i in os.listdir(os.path.join(root_dir)) if os.path.isdir(os.path.join(root_dir, i))]




for dir_name in dir_list:
    # if "LongBox" not in dir_name :continue
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    obj_path    = os.path.join(root_dir, dir_name, "edited", f"{dir_name}.obj")
    usd_path    = os.path.join(root_dir, dir_name, "edited", f"{dir_name}.usd")

    bpy.ops.import_scene.obj(filepath=obj_path)

    
    mesh_objects = [obj for obj in bpy.data.objects if obj.type in ['MESH']]
    empty = [obj for obj in bpy.data.objects if obj.type in ['EMPTY']]
    for emp in empty:
        if len(emp.children) == 0 or len(emp.children) > 1:
            parent_obj = emp
    
    body_list = []
    for obj in mesh_objects:
        if "Body" in obj.name:
            body_list.append(obj)
    if len(body_list) != 2:
        import pdb; pdb.set_trace()
    
    box_obj = body_list[0] if max([i.co.y for i in body_list[0].data.vertices]) > max([i.co.y for i in body_list[1].data.vertices]) else body_list[1]


    scotch_list = []
    for obj in mesh_objects:
        if "Scotch" in obj.name:
            scotch_list.append(obj)
    if len(scotch_list) != 2:
        scotch_list = []
        for obj in mesh_objects:
            if parent_obj.name in obj.name:
                scotch_list.append(obj)
    if len(scotch_list) != 2:
        print("error scotch list != 2")
        import pdb; pdb.set_trace()


    if len(scotch_list[0].data.vertices) > len(scotch_list[1].data.vertices):
        top_bot = scotch_list[0]
        side = scotch_list[1]
    else:
        top_bot = scotch_list[1]
        side = scotch_list[0]

    top_bot.name = "top_bot_Scotch"
    side.name = "side_Scotch"




    top_bot.location.y -= 0.001
    side.location.y -= 0.001



    ##########################################


    bpy.ops.wm.usd_export(filepath=usd_path)
        
    