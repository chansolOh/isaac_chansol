import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument("--output_root_path", type=str, required=False, default="none")
argparse.add_argument("--env_name", type=str, required=False, default="none")
argparse.add_argument("--platform_name", type=str, required=False, default="none")
argparse.add_argument("--random_condition", type=str, required=False,default="none")
argparse.add_argument("--scene_start", type=int, required=False, default=0)
argparse.add_argument("--scene_end", type=int, required=False, default=10)
# args = argparse.parse_args()

def main(output_root_path,
         env_name,
         platform_name,
         random_condition : list,
         scene_start, 
         scene_end, 
        #  object_num = 1, 
         ):

    import sys
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})
    import carb
    print("SceneGen > App_start")
    sys.stdout.flush()
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage


    import omni.isaac.core.prims as Prims
    from omni.isaac.core.utils.rotations import euler_angles_to_quat

    import omni

    import omni.replicator.core as rep
    import omni.graph.core as og
    import omni.kit.commands
    from isaacsim.sensors.camera import Camera


    import numpy as np
    import os
    import json
    import numpy as np

    import carb.settings
    settings = carb.settings.get_settings()
    settings.set("/rtx/useTextureStreaming", False)
    settings.set("/rtx/useAsyncTextureUpload", False)
    settings.set("/rtx/textureCacheSize", 0)


    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    # import pdb; pdb.set_trace()
    from utils.isaac_utils_51 import scan_rep, rep_utils, light_set, sanjabu_Writer
    from utils.general_utils import mat_utils


    ############# set params
    scene_num = scene_start
    render_set = False
    object_path_list = ["/nas/Dataset/Dataset_2026/hanwha"] #"/scan_data_2204/objects_conf.json"


    usd_path = f"/nas/ochansol/isaac/sim2real/hanwha.usd"

    env_conf = {
        "env_name": env_name,
        "platform_name" : "",
        "usd_path": usd_path,
        "position":[0,0,0],
        "orientation":[0,0,0],
        "scale":[1,1,1]
    }
    if len(random_condition)> 0:
        random_condition_name = str(random_condition).strip("[]").replace(", ", "_").replace("'", "")
    else:
        random_condition_name = "default"
    output_path =  f"{output_root_path}/{env_conf['env_name']}/{platform_name}" 

    cam_model_conf_path = "/nas/ochansol/camera_params/azure_kinect_conf_new.json"
    with open(cam_model_conf_path, 'r') as f:
        cam_model_conf = json.load(f)
    cam_conf = {
        "name":"",
        "cam_model_conf_path" : cam_model_conf_path,
        "pixel_size" : cam_model_conf["pixel_size_RGB"]*1000,# 0.0000025, # 2.5um
        "output_size" : (1920,1080),# min object 1920*1280 = 96*54( 5% )
        "clipping_range" : (0.0001, 100000),
        "focus_distance" : 0,
        "f_stop" : 0,
        "cam_poses" : [],
    }

    writer_dict = {
        "rgb"                           : True,
        "bounding_box_2d_loose"         : False,
        "bounding_box_2d_tight"         : True,
        "bounding_box_3d"               : False,
        "distance_to_camera"            : False,
        "distance_to_image_plane"       : True,
        "instance_segmentation"         : True,
        "normals"                       : False,
        "semantic_segmentation"         : False,
        "use_common_output_dir"         : True,
        "pointcloud_include_unlabelled" : False,
        "pointcloud"                    : False,
        "occlusion"                     : False,
    }


    ######################





    my_world = World(stage_units_in_meters=1.0,
                    physics_dt  = 0.001,
                    rendering_dt = 0.05)
    stage = omni.usd.get_context().get_stage()

    my_world.reset()






    ######## env set


    env_usd = add_reference_to_stage(usd_path=env_conf["usd_path"], 
                                        prim_path="/World/"+env_conf["env_name"])

    env_prim = Prims.XFormPrim(name =env_conf["env_name"], prim_path="/World/"+env_conf["env_name"], 
                                position = env_conf["position"], 
                                orientation = mat_utils.euler_to_quat( env_conf["orientation"], degrees = True), 
                                scale = env_conf["scale"] )
    light_list = [i for i in rep_utils.find_lights(env_usd) if "Direct_Lights" in str(i.GetPath()) ]
    Dir_Lights = light_set.Light(light_list)
    light_list = [i for i in rep_utils.find_lights(env_usd) if "ceiling" in str(i.GetPath()) ]
    Ceiling_Lights = light_set.Light(light_list)
    # Lights.random_trans(0.2, [1])
    # Lights.random_exposure()
    # Lights.random_intensity()
    # my_world.reset()

    ###### parent끼리 중복검사 해야됨





    ######object set
    model_list = []
    for path in object_path_list:
        with open(os.path.join(path, "objects_conf.json"),'r'  ) as f:
            model_list += json.load(f)
    



    ######## cam set


    ((fx,_,cx),(_,fy,cy),(_,_,_))= cam_model_conf["intrinsic_matrix"]

    cam_conf["focal_length_isaac"] = (fx+fy)/2*cam_conf["pixel_size"]
    cam_conf["horizontal_aperture"] = cam_conf["output_size"][0]*cam_conf["pixel_size"]
    cam_conf["intrinsic_isaac"] = [[(fx+fy)/2, 0,cam_conf["output_size"][0]/2],
                                [0, (fx+fy)/2, cam_conf["output_size"][1]/2],
                                [0,0,1]]

    # top_view_camera = rep.create.camera(
    #     position = [0,0,1],
    #     rotation = [0,-90,0],
    #     # look_at =obj_rep_list[0].node,
    #     focal_length = cam_conf["focal_length_isaac"], 
    #     focus_distance =cam_conf["focus_distance"], 
    #     f_stop = cam_conf["f_stop"], 
    #     horizontal_aperture = cam_conf["horizontal_aperture"],
    #     clipping_range = cam_conf["clipping_range"])
    

    full_res=(1920,1080)

    full_cam_path = f"{env_usd.GetPrimPath().__str__()}/sim2real_full/Camera"

    top_view_camera = Camera(
        prim_path=full_cam_path,
        name="top_view_camera",
        frequency=20,
        resolution=full_res,)

    top_view_camera.initialize()

    render_product_full = top_view_camera._render_product

    cam_conf1 = cam_conf.copy()
    cam_conf1["name"] = "top_view_camera"

    writer = rep.WriterRegistry.get("SanjabuWriter")
    writer.initialize(
        output_dir=output_path,
        **writer_dict,
    )
    writer.set_path(output_path,
                    rgb_path = "rgb",
                    bounding_box_path = "bbox",
                    distance_to_image_plane_path = "depth",
                    instance_segmentation_path = "inst_seg",
                    pointcloud_path = "pointcloud",
                    normals_path = "normals",
                    occlusion_path = "occlusion",)
    writer.set_cam_name_list([cam_conf1["name"],])# cam_conf2["name"]])

    # # Attach render_product to the writer

    # instance_seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
    # instance_seg_annotator.attach([render_product_top])
    # depth_cam_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    # depth_cam_annotator.attach([render_product])
    # depth_plane_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    # depth_plane_annotator.attach([render_product])

    writer.attach([render_product_full,])# render_product_side])
    rep.orchestrator.pause()
    rep.orchestrator.set_capture_on_play(False)

    print("render set complete ")


    ##################################################################################33
    my_world.reset()
    my_world.stop()
    # writer.set_frame(frame_id=0)
    os.makedirs(os.path.join(output_path,random_condition_name,"conf"), exist_ok=True)
    print("dir making complete : ")

    physics_scene_conf={
        # 'physxScene:enableGPUDynamics': 1, # True
        # 'physxScene:broadphaseType' : "GPU",
        # 'physxScene:collisionSystem' : "PCM",
        
        # 'physxScene:timeStepsPerSecond' : 1000,
        'physxScene:minPositionIterationCount' : 30,
        'physxScene:minVelocityIterationCount' : 1,
        # "physics:gravityMagnitude":35,
        # "physxScene:updateType":"Asynchronous",
    }
    for key in physics_scene_conf.keys():
        stage.GetPrimAtPath("/physicsScene").GetAttribute(key).Set(physics_scene_conf[key])
        
        
    platform_area_prims = rep_utils.find_target_name(env_prim.prim,["Mesh"],"platform_area")
    platform_area_prims = [i.GetParent() for i in platform_area_prims if i.GetParent().GetName() == platform_name][0]


    platform_path = platform_area_prims.GetPath().__str__()
    platform_rep = scan_rep.Scan_Rep_Platform(prim_path = platform_path,scale = [1,1,1], class_name = platform_path.split("/")[-1])
   

    my_world.reset()
  
    platform_tf = rep_utils.find_parents_tf(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
    platform_scale = rep_utils.find_parents_scale(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
    platform_rep.set_tf(platform_tf)
    platform_rep.set_scale(platform_scale)

    # cam_pos_prim = rep_utils.find_target_name(platform_area_prims,["Xform"],"cam_pos")[0]
    # cam_tf = rep_utils.find_parents_tf(cam_pos_prim, include_self=True)
    # prs = rep_utils.tf_to_pos_rot_scale(cam_tf)
    # rep_utils.set_node_pose(top_view_camera, prs["position"] , rotation = [0,-90,0])
    # print("platform set complete")

    sdg_pipe_prim = stage.GetPrimAtPath("/Replicator/SDGPipeline")
    sdg_pipe_children = sdg_pipe_prim.GetChildren()

    def remove_all_objects(obj_rep_all_list, sdg_pipe_prim, sdg_pipe_children):
        for OBJ in obj_rep_all_list:
            og.GraphController.delete_node(OBJ.node.node.get_prim_path())
            stage.RemovePrim(OBJ.prim.GetPath())
        
        for prim in sdg_pipe_prim.GetChildren():
            if prim not in sdg_pipe_children:
                stage.RemovePrim(prim.GetPath())



    import time
    print("SceneGen > reset_complete")
    sys.stdout.flush()
    while scene_num<=scene_end:
        print("SceneGen > START")
        sys.stdout.flush()
        print(f"SceneGen > SCENE:{scene_num}")
        sys.stdout.flush()
        data_gen_time = time.time()
        print("####################    scene_num : ",scene_num)
        settings.set("/rtx/rendermode", "RayTraced")
        scene_name = f"{scene_num:04d}"
        

        print("platform_rep : ", platform_rep.prim)


        obj_rep_all_list = [platform_rep]
        # size_rank = np.random.randint(0, 2) # 0 or 1 
        # size_rank=2
        # print("size_rank : ", size_rank)

        profile_num = np.random.randint(0,10)
        samick_box_num = np.random.randint(0,20)
        sampled_model_dict = {}
        sampled_model_list = []
        for model_attr in model_list:
            if model_attr["name"] == "samick_box":
                sampled_model_list+=[model_attr["name"]]*samick_box_num
            else:
                sampled_model_list+=[model_attr["name"]]*profile_num

            sampled_model_dict[model_attr["name"]] = model_attr


        for model_attr in sampled_model_list:
            model_attr = sampled_model_dict[model_attr]
            print("model_attr : ", model_attr["name"])
            scan_obj = scan_rep.Scan_Rep(usd_path =  model_attr["path"],
                                    class_name = model_attr["name"],
                                    size = 0,)

            obj_rep_all_list.append(scan_obj)
        
        # ##### @@@@@@@@@@@@@ debuging specific object
        # model_attr = [i for i in model_list if i["name"] == "whiteboard_eraser"][0]
        # scan_obj = scan_rep.Scan_Rep(usd_path =  model_attr["path"],
        #                 class_name = model_attr["name"],
        #                 size = model_attr["size_rank"],)
        # obj_rep_all_list.append(scan_obj)
        # ##########################

        for OBJ in obj_rep_all_list[1:]:
            print("set collider for : ", OBJ.class_name)
            OBJ.set_rigidbody_collider()
            # OBJ.set_contact_sensor()
            OBJ.set_physics_material(
                dynamic_friction=0.25,
                static_friction=0.4,
                restitution=0.0
            )



        my_world.reset()
        my_world.stop()


        Dir_Lights.random_exposure(val = 0.3)#, default_exposure = np.random.uniform(1,2.3) )
        Dir_Lights.random_temp(val = 300, default_temp = 5800)
        Dir_Lights.random_trans(val=1.3,fixed_axis=[2])
        Ceiling_Lights.random_exposure(val = 1.1)#, default_exposure = np.random.uniform(1,2.3) )
        Ceiling_Lights.random_temp(val = 300, default_temp = 5800)

        rep_utils.scatter_in_platform_area(obj_rep_all_list[0],obj_rep_all_list[1:],fixed_first = False)

        obj_rep_list = obj_rep_all_list[1:]
        
        my_world.play()
        obj_rotation_buf = []
        obj_location_buf = []

        for i in range(20):
            my_world.step(render = render_set)
            obj_rotation_buf.append([obj.get_local_pose()["rotation"]for obj in obj_rep_list])
            obj_location_buf.append([obj.get_local_pose()["translation"] for obj in obj_rep_list])

        while True:
            my_world.step(render = render_set)
            del(obj_rotation_buf[0])
            del(obj_location_buf[0])
            obj_rotation_buf.append([obj.get_local_pose()["rotation"]for obj in obj_rep_list])
            obj_location_buf.append([obj.get_local_pose()["translation"] for obj in obj_rep_list])
            # print(np.array(obj_rotation_buf).std(axis=0).max())
            # print(np.array(obj_location_buf).std(axis=0).max())
            if np.array(obj_rotation_buf).std(axis=0).max()<=0.00001 and np.array(obj_location_buf).std(axis=0).max()<=0.0001:
                break
            
            if my_world.current_time>6:
                break

        print("current_time : ",my_world.current_time)
        
        

        ########  
        obb_list = []
        for obj in obj_rep_list:
            obb_list.append(obj.get_obb())

        obb_arr = np.vstack(obb_list)
        obb_min = obb_arr.min(axis=0)
        obb_max = obb_arr.max(axis=0)
        center = (obb_min+obb_max)/2

        ########
    
        if "cam_random_position" in random_condition:
            top_view_camera.set_local_pose(translation=[center[0],center[1],center[2]+1.2,]) #orientation=[1,0,0,0])
        rep.orchestrator.step()


        # if writer.get_data()["annotators"]["instance_segmentation_fast"]["Replicator"]["idToSemantics"].keys().__len__()<2+object_num:
        #     print("scene_reset, 탑뷰 카메라 오류")
        #     remove_all_objects(obj_rep_list, sdg_pipe_prim, sdg_pipe_children)
        #     continue

        # for _ in range(8):
        #     with side_view_camera:
        #         rad  = np.random.randint(0,360)/180*np.pi
        #         dist = 0.8
        #         x,y,z = dist*np.cos(rad)+center[0], dist*np.sin(rad)+center[1], center[2]+1
        #         rep.modify.pose(position=(x,y,z),
        #                         look_at = center,)
        #     rep.orchestrator.step()
    


        #     side_view_obj_count = writer.get_data()["annotators"]["instance_segmentation_fast"]["Replicator_01"]["idToSemantics"].keys().__len__()
        #     if side_view_obj_count<7:
        #         print("side_view_obj_count : ", side_view_obj_count)
        #         continue
        #     else:
        #         break
        # if side_view_obj_count<7:
        #     remove_all_objects(obj_rep_list, sdg_pipe_prim, sdg_pipe_children)
        #     continue

        # side_view_bboxes = np.array(writer.get_data()["annotators"]["bounding_box_2d_tight_fast"]["Replicator_01"]["data"].tolist())[:,1:5]
        # side_view_bboxes_xmax = np.max(side_view_bboxes[:,2])>=cam_conf["output_size"][0]
        # side_view_bboxes_ymax = np.max(side_view_bboxes[:,3])>=cam_conf["output_size"][1]
        # side_view_bboxes_min = np.min(side_view_bboxes)<=0
        # if side_view_bboxes_xmax or side_view_bboxes_ymax or side_view_bboxes_min:
        #     continue

        my_world.pause()
        writer.set_frame(frame_id=scene_num)
            ####
        # rep.orchestrator.run()
        # rep.orchestrator.step()
        # rep.orchestrator.pause()

        settings.set("/rtx/rendermode", "PathTracing")

        settings.set("/rtx/pathtracing/spp", 32) 
        settings.set("/rtx/pathtracing/totalSpp", 160)
        settings.set("/rtx/pathtracing/maxBounces", 12)
        settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 12)


        writer.output_path = os.path.join(output_path,random_condition_name)
        rep.orchestrator.step()

        print("spp complete")
        

        obj_conf = []

        for OBJ in obj_rep_list:
            pose = OBJ.get_world_pose()
            scale = OBJ.get_scale()
            obj_conf.append({
                "class" : OBJ.class_name,
                "usd_path" : OBJ.usd_path,
                "translate" : pose["translation"],
                "orient" : pose["rotation"],
                "scale" : scale,
            })
            
        cam_conf1["cam_poses"] = np.array(rep_utils.cal_cam_tf(top_view_camera.prim)).T.tolist()
        # cam_conf2["cam_poses"] = np.array(rep_utils.cal_cam_tf(side_view_camera)).T.tolist()
        cam_conf_list = [ cam_conf1,]# cam_conf2 ]

        Lights_conf = Dir_Lights.get_all_state()
        
        platform_rep.usd_path = "/nas/ochansol/isaac/sim2real/hanwha.usd"
        
        platform_conf = {
            "name": platform_rep.class_name,
            "usd_path": platform_rep.usd_path,
            "translate": env_conf["position"],
            "orient": euler_angles_to_quat(env_conf["orientation"], degrees=True).tolist(),
            "scale": env_conf["scale"],
            
        }


        save_conf = {
            "envs": env_conf,
            "objects" : obj_conf,
            "platform" :platform_conf,
            "cameras" : cam_conf_list,
            "lights" : Lights_conf,
            "physics_scene" : physics_scene_conf,
        }




        with open(writer.output_path+f"/conf/{scene_name}.json", 'w') as f:
            json.dump(save_conf, f, indent=4)
        
        
        scene_num+=1

        print("scene save complete : ", scene_name)
        


        remove_all_objects(obj_rep_list, sdg_pipe_prim, sdg_pipe_children)

        print("SceneGen > data_gen_time : ", time.time()-data_gen_time)
        sys.stdout.flush()
    print("SceneGen > END")
    sys.stdout.flush()
    simulation_app.close()

if __name__ == "__main__":
    args = argparse.parse_args()
    output_root_path = args.output_root_path if args.output_root_path!="none" else "/nas/Dataset/Dataset_2026"
    env_name = args.env_name if args.env_name!="none" else "hanwha"
    platform_name = args.platform_name if args.platform_name!="none" else "chicken_box"
    random_condition = args.random_condition.strip("[]").split(",") if args.random_condition!="none" else []
    scene_start = args.scene_start if args.scene_start!=0 else 0
    scene_end = args.scene_end if args.scene_end!=10 else scene_start+2000
    # object_num = args.object_num

    main(
        # output_root_path = "/nas/Dataset/Dataset_2025",
        output_root_path = output_root_path,
        env_name = env_name,
        platform_name = platform_name,
        random_condition = random_condition,
        scene_start = scene_start,
        scene_end = scene_end,
        # object_num = 1,
    )
