import omni.replicator.core as rep
import omni.usd
from omni.physx.scripts import utils as physx_utils
import omni.isaac.core.utils.bounds as bounds_utils
from omni.isaac.core.utils.rotations import euler_angles_to_quat, euler_to_rot_matrix, quat_to_rot_matrix

from typing import Callable, Dict, List, Tuple, Union
from pxr import Sdf, Tf, Usd, UsdGeom, UsdShade, Gf, PhysxSchema, UsdPhysics
from omni.isaac.debug_draw import _debug_draw
import numpy as np
import carb

import my_rep
from omni.isaac.sensor import _sensor
import os

from omni.physx import get_physx_interface, get_physx_simulation_interface
from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts.physicsUtils import *
from omni.isaac.core.materials.physics_material import PhysicsMaterial
import omni.isaac.core.utils.rotations as rot_utils
import cs_rep_utils as csr



class Box_Rep(my_rep.rep_usd):
    
    def __init__(self,
                class_name,
                width,
                height,
                depth,
                split = False,
                usd_path = None,
                prim_path = "",
                position = [0,0,0], 
                rotation = [0,0,0], 
                scale=[0.1,0.1,0.1],
                semantics: List[Tuple[str, str]] = None, 
                visible = True,
                ) :

        super().__init__(prim_path="object/"+class_name if prim_path == "" else prim_path+"/" + class_name, 
                        usd_path= usd_path, 
                        semantics=None,
                        count=1, 
                        rigidbody_collider=False,
                        particle_cloth=False,
                        )
        
        self.usd_path = usd_path
        self.class_name = class_name
        self.width = width
        self.height = height
        self.depth = depth
        self.stage = omni.usd.get_context().get_stage()
        self.prim = self.get_prims()[0]
        self.obb = self.get_init_obb()
        self.contact_prim_path = []
        self.contact_prim_state = 0
        
        self.box_meshes = csr.find_targets(self.prim,["Mesh"])
        self.box_geom = csr.find_targets(self.prim,["GeomSubset"])
        
        self.scotch_meshes = []
        for mesh in self.box_meshes:
            if "Scotch" in mesh.GetName():
                self.scotch_meshes.append(mesh)
        self.body = []
        for mesh in self.box_geom:
            if "Cardboard" in mesh.GetName():
                self.body.append(mesh)
                
                
        with self.node:
            rep.modify.visibility(visible)
        with self.node:
            rep.modify.pose(position=position, rotation=rotation, scale=scale)
        


        xforms = csr.find_targets(self.prim, "Xform")

        for xform in xforms:
            if xform.GetName() in "Side":
                self.side_xform = xform
            elif xform.GetName() in "Top_Bot":
                self.top_bot_xform = xform
            # elif xform.GetName() in "Body":
            #     self.Body_xform = xform
                
        if split:
            try:
                self.side_node = rep.create.group([self.side_xform.GetPath()])
                
                self.top_bot_node = rep.create.group([self.top_bot_xform.GetPath()])
            except:
                print(self.class_name)
            self.set_semantic(self.side_node,"class","side")
            self.set_semantic(self.top_bot_node,"class","top_bot")
         
        self.set_semantic(self.node,"class",self.class_name )
        
        self.meshes = csr.find_targets(self.prim, "Mesh")
        self.mesh_parent_path = self.meshes[0].GetParent().GetPath().pathString
        

    def set_semantic(self,node, class_name, type_name):
        rep.utils._set_semantics(node, [(class_name,type_name)])

    def set_contact_sensor(self):
        success, _isaac_sensor_prim = omni.kit.commands.execute(
            "IsaacSensorCreateContactSensor",
            path="Contact_Sensor",
            parent = self.mesh_parent_path,
            sensor_period=1,
            min_threshold=0.0001,
            max_threshold=100000,
            translation = Gf.Vec3d(0, 0, 0),
        )
        

        self.contact_prim = self.stage.GetPrimAtPath(self.mesh_parent_path)
        self.contact_report = PhysxSchema.PhysxContactReportAPI.Apply(self.contact_prim)
        self.contact_report.CreateThresholdAttr().Set(0)

        self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
        self._contact_sensor_path = os.path.join(self.mesh_parent_path, "Contact_Sensor")
        self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)

    def set_collider(self, collider_type="convexDecomposition"):
        for prim in self.get_prims():
            physx_utils.setStaticCollider(prim, collider_type)
            for mesh in self.meshes:
                mesh.GetAttribute('physxCollision:contactOffset').Set(0.000001)
                mesh.GetAttribute('physxCollision:restOffset').Set(0)
                mesh.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Set(True)

            
    def set_rigidbody_collider(self):
        for prim in self.get_prims():
            # import pdb;pdb.set_trace()
            set_list = []
            for mesh in self.meshes:
                if mesh.HasAttribute("physxConvexDecompositionCollision:maxConvexHulls"):
                    set_list.append({
                            "maxConvexHulls":mesh.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Get(),
                            "hullVertexLimit":mesh.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Get(),
                            "voxelResolution":mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Get(),
                            "errorPercentage":mesh.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Get(),
                        })
                else:
                    set_list.append({
                            "maxConvexHulls":240,
                            "hullVertexLimit":64,
                            "voxelResolution":700000,
                            "errorPercentage":8,
                        })
            physx_utils.setRigidBody(prim, "convexDecomposition", False)
            # physx_utils.setRigidBody(prim, "meshSimplification", False)
            # physx_utils.setRigidBody(prim, "sdf", False)
            prim.GetAttribute("physxRigidBody:maxAngularVelocity").Set(720)
            prim.GetAttribute("physxRigidBody:maxLinearVelocity").Set(2.5)
            prim.GetAttribute("physxRigidBody:linearDamping").Set(0.7)
            prim.GetAttribute("physxRigidBody:enableCCD").Set(True)
        for mesh, settings in zip(self.meshes, set_list):
            mesh.GetAttribute('physxCollision:contactOffset').Set(0.000001)
            mesh.GetAttribute('physxCollision:restOffset').Set(0)
            mesh.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Set(True)
            # mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Set(600000)
            mesh.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Set(settings["maxConvexHulls"])
            mesh.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Set(settings["hullVertexLimit"])
            mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Set(settings["voxelResolution"])
            mesh.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Set(settings["errorPercentage"])

    def set_physics_material(self, dynamic_friction=0.25, static_friction=0.4, restitution=0.0):
        if  not self.stage.GetPrimAtPath(os.path.join(self.mesh_parent_path, "PhysicsMaterial")).IsValid():
            physics_material = PhysicsMaterial(
                prim_path=os.path.join(self.mesh_parent_path, "PhysicsMaterial"), 
                dynamic_friction=dynamic_friction, 
                static_friction=static_friction, 
                restitution=restitution
            )

            for mesh in self.meshes:
                add_physics_material_to_prim(self.stage, mesh, physics_material.prim.GetPath())
                UsdPhysics.MassAPI.Apply(mesh)
        # import pdb;pdb.set_trace()
        # physics_material_path = Sdf.Path(os.path.join(self.get_prims()[0].GetPath().pathString, "PhysicsMaterial"))
        # material = self.stage.DefinePrim(physics_material_path, "PhysxMaterial")
        # physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material)

    def set_pose(self, position, rotation):
        self.init_position = position
        self.init_rotation = rotation
        with self.node:
            rep.modify.pose(position = position, rotation = rotation)
    def set_width_to_height(self):
        pose = self.get_pose()
        position = np.array(pose["translation"]) + np.array([0,self.width,0])
        rotation = np.array(pose["rotation"]) + np.array([0,0,-90])
        with self.node:
            rep.modify.pose(position = position, rotation = rotation)
    def set_scale(self, scale):
        with self.node:
            rep.modify.pose(scale = scale)
    def set_visible(self, visible):
        with self.node:
            rep.modify.visibility(visible)
    
    def set_change_scotch_mat(self,mat):
        for mesh in self.scotch_meshes:
            UsdShade.MaterialBindingAPI(mesh).Bind(mat)
            
    def set_change_body_mat(self, mat):
        for mesh in self.body:
            UsdShade.MaterialBindingAPI(mesh).Bind(mat)

    def get_pose(self):
        trans = self.prim.GetAttribute("xformOp:translate").Get()
        if self.prim.HasAttribute("xformOp:rotateXYZ"):
            rot = self.prim.GetAttribute("xformOp:rotateXYZ").Get()
            return {
                "translation" :np.array(trans).tolist(),
                "rotation" : np.array(rot).tolist()
            }
        elif self.prim.HasAttribute("xformOp:orient"):
            rot = self.prim.GetAttribute("xformOp:orient").Get()
            return {
                    "translation" :np.array(trans).tolist(),
                    "rotation" : rot_utils.gf_quat_to_np_array(rot).tolist()
            }
            
    def get_scale(self):
        if self.prim.HasAttribute("xformOp:scale"):
            return np.array(self.prim.GetAttribute("xformOp:scale").Get()).tolist()

    def get_init_obb(self):
        cache = bounds_utils.create_bbox_cache()
        obb = np.array(bounds_utils.compute_obb_corners(cache,self.prim.GetPath()))
        return obb
    
    def get_obb(self):
        tf = np.array(csr.find_parents_tf(self.prim, include_self=True)).T
        return tf.dot(np.vstack((self.obb.T, np.ones(len(self.obb))))).T[:,:3]
    
    
    
    
    
    
    
    
    
class Box_In_Pal_Rep():
    
    def __init__(self,prim
                ) :
        
        self.prim = prim    
        xforms = csr.find_targets(self.prim, "Xform")


        
        self.box_meshes = csr.find_targets(self.prim,["Mesh"])
        self.box_geom = csr.find_targets(self.prim,["GeomSubset"])
        
        self.scotch_meshes = []
        for mesh in self.box_meshes:
            if "Scotch" in mesh.GetName():
                self.scotch_meshes.append(mesh)
        self.body = []
        for mesh in self.box_geom:
            if "Cardboard" in mesh.GetName():
                self.body.append(mesh)
                

        
        for xform in xforms:
            if xform.GetName() in "Side":
                self.side_xform = xform
            elif xform.GetName() in "Top_Bot":
                self.top_bot_xform = xform
            # elif xform.GetName() in "Body":
            #     self.Body_xform = xform
        self.body_xform = self.top_bot_xform.GetParent().GetParent()
                
        
        self.side_node = rep.create.group([self.side_xform.GetPath()])
        
        self.top_bot_node = rep.create.group([self.top_bot_xform.GetPath()])
        self.body_node = rep.create.group([self.body_xform.GetPath()])
        self.set_semantic(self.side_node,"class","side")
        self.set_semantic(self.top_bot_node,"class","top_bot")
        self.set_semantic(self.body_node,"class",self.body_xform.GetName() )
         
        # self.set_semantic(self.node,"class",self.class_name )
        
        

    def set_semantic(self,node, class_name, type_name):
        rep.utils._set_semantics(node, [(class_name,type_name)])
        
    def set_change_scotch_mat(self,mat):
        for mesh in self.scotch_meshes:
            UsdShade.MaterialBindingAPI(mesh).Bind(mat)
            
    def set_change_body_mat(self, mat):
        for mesh in self.body:
            UsdShade.MaterialBindingAPI(mesh).Bind(mat)






class Box_Rep_No_Split(my_rep.rep_usd):
    
    def __init__(self,
                class_name,
                width,
                height,
                depth,
                split = False,
                usd_path = None,
                prim_path = "",
                position = [0,0,0], 
                rotation = [0,0,0], 
                scale=[0.1,0.1,0.1],
                semantics: List[Tuple[str, str]] = None, 
                visible = True,
                ) :

        super().__init__(prim_path="object/"+class_name if prim_path == "" else prim_path+"/" + class_name, 
                        usd_path= usd_path, 
                        semantics=None,
                        count=1, 
                        rigidbody_collider=False,
                        particle_cloth=False,
                        )
        
        self.usd_path = usd_path
        self.class_name = class_name
        self.width = width
        self.height = height
        self.depth = depth
        self.stage = omni.usd.get_context().get_stage()
        self.prim = self.get_prims()[0]
        self.obb = self.get_init_obb()
        self.contact_prim_path = []
        self.contact_prim_state = 0
        

                
        with self.node:
            rep.modify.visibility(visible)
        with self.node:
            rep.modify.pose(position=position, rotation=rotation, scale=scale)
        
         
        self.set_semantic(self.node,"class",self.class_name )
        
        self.meshes = csr.find_targets(self.prim, ["Mesh"])
        self.mesh_parent_path = self.meshes[0].GetParent().GetPath().pathString
        

    def set_semantic(self,node, class_name, type_name):
        rep.utils._set_semantics(node, [(class_name,type_name)])

    def set_contact_sensor(self):
        success, _isaac_sensor_prim = omni.kit.commands.execute(
            "IsaacSensorCreateContactSensor",
            path="Contact_Sensor",
            parent = self.mesh_parent_path,
            sensor_period=1,
            min_threshold=0.0001,
            max_threshold=100000,
            translation = Gf.Vec3d(0, 0, 0),
        )
        

        self.contact_prim = self.stage.GetPrimAtPath(self.mesh_parent_path)
        self.contact_report = PhysxSchema.PhysxContactReportAPI.Apply(self.contact_prim)
        self.contact_report.CreateThresholdAttr().Set(0)

        self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
        self._contact_sensor_path = os.path.join(self.mesh_parent_path, "Contact_Sensor")
        self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)

    def set_collider(self, collider_type="convexDecomposition"):
        for prim in self.get_prims():
            physx_utils.setStaticCollider(prim, collider_type)
            for mesh in self.meshes:
                mesh.GetAttribute('physxCollision:contactOffset').Set(0.000001)
                mesh.GetAttribute('physxCollision:restOffset').Set(0)
                mesh.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Set(True)

            
    def set_rigidbody_collider(self):
        for prim in self.get_prims():
            # import pdb;pdb.set_trace()
            set_list = []
            for mesh in self.meshes:
                if mesh.HasAttribute("physxConvexDecompositionCollision:maxConvexHulls"):
                    set_list.append({
                            "maxConvexHulls":mesh.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Get(),
                            "hullVertexLimit":mesh.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Get(),
                            "voxelResolution":mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Get(),
                            "errorPercentage":mesh.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Get(),
                        })
                else:
                    set_list.append({
                            "maxConvexHulls":240,
                            "hullVertexLimit":64,
                            "voxelResolution":700000,
                            "errorPercentage":8,
                        })
            physx_utils.setRigidBody(prim, "convexDecomposition", False)
            # physx_utils.setRigidBody(prim, "meshSimplification", False)
            # physx_utils.setRigidBody(prim, "sdf", False)
            prim.GetAttribute("physxRigidBody:maxAngularVelocity").Set(720)
            prim.GetAttribute("physxRigidBody:maxLinearVelocity").Set(2.5)
            prim.GetAttribute("physxRigidBody:linearDamping").Set(0.7)
            prim.GetAttribute("physxRigidBody:enableCCD").Set(True)
        for mesh, settings in zip(self.meshes, set_list):
            mesh.GetAttribute('physxCollision:contactOffset').Set(0.000001)
            mesh.GetAttribute('physxCollision:restOffset').Set(0)
            mesh.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Set(True)
            # mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Set(600000)
            mesh.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Set(settings["maxConvexHulls"])
            mesh.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Set(settings["hullVertexLimit"])
            mesh.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Set(settings["voxelResolution"])
            mesh.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Set(settings["errorPercentage"])

    def set_physics_material(self, dynamic_friction=0.25, static_friction=0.4, restitution=0.0):
        if  not self.stage.GetPrimAtPath(os.path.join(self.mesh_parent_path, "PhysicsMaterial")).IsValid():
            physics_material = PhysicsMaterial(
                prim_path=os.path.join(self.mesh_parent_path, "PhysicsMaterial"), 
                dynamic_friction=dynamic_friction, 
                static_friction=static_friction, 
                restitution=restitution
            )

            for mesh in self.meshes:
                add_physics_material_to_prim(self.stage, mesh, physics_material.prim.GetPath())
                UsdPhysics.MassAPI.Apply(mesh)
        # import pdb;pdb.set_trace()
        # physics_material_path = Sdf.Path(os.path.join(self.get_prims()[0].GetPath().pathString, "PhysicsMaterial"))
        # material = self.stage.DefinePrim(physics_material_path, "PhysxMaterial")
        # physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material)

    def set_pose(self, position, rotation):
        self.init_position = position
        self.init_rotation = rotation
        with self.node:
            rep.modify.pose(position = position, rotation = rotation)
    def set_width_to_height(self):
        pose = self.get_pose()
        position = np.array(pose["translation"]) + np.array([0,self.width,0])
        rotation = np.array(pose["rotation"]) + np.array([0,0,-90])
        with self.node:
            rep.modify.pose(position = position, rotation = rotation)
    def set_scale(self, scale):
        with self.node:
            rep.modify.pose(scale = scale)
    def set_visible(self, visible):
        with self.node:
            rep.modify.visibility(visible)
            

    def get_pose(self):
        trans = self.prim.GetAttribute("xformOp:translate").Get()
        if self.prim.HasAttribute("xformOp:rotateXYZ"):
            rot = self.prim.GetAttribute("xformOp:rotateXYZ").Get()
            return {
                "translation" :np.array(trans).tolist(),
                "rotation" : np.array(rot).tolist()
            }
        elif self.prim.HasAttribute("xformOp:orient"):
            rot = self.prim.GetAttribute("xformOp:orient").Get()
            return {
                    "translation" :np.array(trans).tolist(),
                    "rotation" : rot_utils.gf_quat_to_np_array(rot).tolist()
            }
            
    def get_scale(self):
        if self.prim.HasAttribute("xformOp:scale"):
            return np.array(self.prim.GetAttribute("xformOp:scale").Get()).tolist()

    def get_init_obb(self):
        cache = bounds_utils.create_bbox_cache()
        obb = np.array(bounds_utils.compute_obb_corners(cache,self.prim.GetPath()))
        return obb
    
    def get_obb(self):
        tf = np.array(csr.find_parents_tf(self.prim, include_self=True)).T
        return tf.dot(np.vstack((self.obb.T, np.ones(len(self.obb))))).T[:,:3]
    