import numpy as np
import matplotlib.pyplot as plt

from . import mat_utils

def cal_grasp_width_z(x,y,angle, width, depth_img, cam_tf = None):
    gripper_height = 0.02
    gripper_depth = 0.02
    finger_thickness = 0.01
    # focal_length = 957.22335
    # focal_length = np.array([960,954.4467])
    focal_l_x = 960.0
    focal_l_y = 954.4467
    cx = 960
    cy = 540
    cam_tf = cam_tf if cam_tf is not None else mat_utils.trans([-0.328,0.39,2.046]).dot(mat_utils.rot_z(90)).round(3)

    return main(x,y,angle, width, depth_img, gripper_height, gripper_depth, finger_thickness, focal_l_x, focal_l_y,cx,cy, cam_tf)





def main(x,y,angle, width, depth_img, gripper_height, gripper_depth, finger_thickness, focal_l_x, focal_l_y,cx,cy, cam_tf):
    grasp_bbox_x = x
    grasp_bbox_y = y
    grasp_bbox_deg = angle+90
    grasp_bbox_width = width
    grasp_bbox_height = 20

    grasp_bbox = np.array([[ grasp_bbox_height/2, -grasp_bbox_width/2],
                                [ grasp_bbox_height/2,  grasp_bbox_width/2],
                                [-grasp_bbox_height/2,  grasp_bbox_width/2],
                                [-grasp_bbox_height/2, -grasp_bbox_width/2]]).T

    grasp_bbox = np.vstack((grasp_bbox,np.zeros_like(grasp_bbox[0])))
    grasp_bbox_rotated = mat_utils.rot_z(grasp_bbox_deg).dot(np.vstack((grasp_bbox,np.ones_like(grasp_bbox[0]))))[:2].astype(int)
    grasp_bbox_rotated += np.array([[grasp_bbox_y,grasp_bbox_x]]).T


    depth_y, depth_x = np.where(depth_img == depth_img)
    idx_in_bbox = mat_utils.select_points(grasp_bbox_rotated[None,:],np.array([depth_y,depth_x]))
    world_z = np.min(depth_img[depth_y[idx_in_bbox],depth_x[idx_in_bbox]])
    # world_z_one = depth_img[int(grasp_bbox_y),int(grasp_bbox_x)] ### center 말고 bbox 안의 depth 평균으로 수정
    world_bbox = (grasp_bbox_rotated-np.array([[cy,cx]]).T)/np.array([focal_l_x,focal_l_y])[:,None]*world_z
    world_bbox = world_bbox.T[:2]
    world_width = np.sqrt(np.sum((world_bbox[0]-world_bbox[1])**2,axis = 0))


    grasp_bbox_x = (grasp_bbox_x-cx)/focal_l_x*world_z
    grasp_bbox_y = (grasp_bbox_y-cy)/focal_l_y*world_z
    mouse_pt = np.array([[grasp_bbox_x, grasp_bbox_y, world_z, 1]]).T
    world_to_mouse = np.array(cam_tf).dot(mat_utils.rot_x(180)).dot(mouse_pt)[:3]




    sample_rate = 0.5

    ###### point cloud sampling
    IDX = np.argwhere(depth_img==depth_img).T
    Z_all = depth_img[IDX[0],IDX[1]]
    sample_idx = np.random.choice(len(Z_all),int(len(Z_all)*sample_rate),replace=False)
    SAMPLED_IDX = IDX[:,sample_idx]
    Z_sample = Z_all[sample_idx]
    X_sample = (SAMPLED_IDX[1]-cx)/focal_l_x*Z_sample
    Y_sample = (SAMPLED_IDX[0]-cy)/focal_l_y*Z_sample
    # PCD = np.array([X_sample,Y_sample,Z_sample,np.ones_like(X_sample)])
    cam_to_pt_all = np.array([X_sample,Y_sample,Z_sample,np.ones_like(X_sample)])
    PCD = np.array(cam_tf).dot(mat_utils.rot_x(180)).dot(cam_to_pt_all)[:3]




    world_mouse_idx = np.argmin(np.sqrt(np.sum((world_to_mouse[:2]-PCD[:2])**2,axis = 0)))
    world_to_grasp = np.array([PCD[:,world_mouse_idx]])


    gripper_finger_bbox = np.array([[ gripper_height/2, -world_width/2 -  finger_thickness],
                                    [ gripper_height/2, -world_width/2 ],
                                    [-gripper_height/2, -world_width/2 ],
                                    [-gripper_height/2, -world_width/2 -  finger_thickness],

                                    [ gripper_height/2,  world_width/2 ],
                                    [ gripper_height/2,  world_width/2 +  finger_thickness],
                                    [-gripper_height/2,  world_width/2 +  finger_thickness],
                                    [-gripper_height/2,  world_width/2 ]])
    ## 우측 하단 꼭지에서 반시계 방향
    gripper_width_bbox = np.array([[ gripper_height/2, -world_width/2],
                                [ gripper_height/2,  world_width/2],
                                [-gripper_height/2,  world_width/2],
                                [-gripper_height/2, -world_width/2]]) 

    gripper_center_point = np.array([[ 0,0 ]])
    gripper_bbox = np.vstack((gripper_finger_bbox,gripper_width_bbox,gripper_center_point)).T
    gripper_bbox = np.vstack((gripper_bbox,np.zeros_like(gripper_bbox[0])))




    # print("gripper_rot : ",grasp_bbox_deg)
    gripper_bbox_3d = mat_utils.rot_z(grasp_bbox_deg).dot(np.vstack((gripper_bbox,np.ones_like(gripper_bbox[0]))))[:3]
    gripper_bbox_3d = np.tile(world_to_grasp[...,None], (1,1,gripper_bbox_3d.shape[1])) + np.tile(gripper_bbox_3d[None,...],(len(world_to_grasp),1,1))
    bbx_xmin, bbx_ymin, bbx_xmax, bbx_ymax =   [np.min(gripper_bbox_3d[:,0,:]),
                                                np.min(gripper_bbox_3d[:,1,:]),
                                                np.max(gripper_bbox_3d[:,0,:]),
                                                np.max(gripper_bbox_3d[:,1,:])]



    X_all = (SAMPLED_IDX[1]-cx)/focal_l_x*depth_img[SAMPLED_IDX[0],SAMPLED_IDX[1]]
    Y_all = (SAMPLED_IDX[0]-cy)/focal_l_y*depth_img[SAMPLED_IDX[0],SAMPLED_IDX[1]]
    cam_to_obj_part = np.array([X_all,Y_all,depth_img[SAMPLED_IDX[0],SAMPLED_IDX[1]],np.ones_like(X_all)])
    world_to_obj_part = np.array(cam_tf).dot(mat_utils.rot_x(180)).dot(cam_to_obj_part)[:3]
    world_to_obj_part_idx = np.where( (world_to_obj_part[0]>bbx_xmin) &(world_to_obj_part[0]<bbx_xmax) & (world_to_obj_part[1]>bbx_ymin) & (world_to_obj_part[1]<bbx_ymax) )[0]
    world_to_obj_part = world_to_obj_part[:,world_to_obj_part_idx]

    #random sampling
    point_in_bboxes_idx = mat_utils.select_points(gripper_bbox_3d,world_to_obj_part)




    for bbox_num in range(len(gripper_bbox_3d)):

        finger_max_depth = np.max([0 if len(point_in_bboxes_idx[bbox_num][0])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][0]].max(),  
                                    0 if len(point_in_bboxes_idx[bbox_num][1])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][1]].max()])
        palm_max_depth   = np.max( 0 if len(point_in_bboxes_idx[bbox_num][2])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][2]].max())

        ##############
        crit = palm_max_depth - finger_max_depth
        # print(crit)
        margin = 0.002 # default = 0.005

        if crit<0.01:
            print("not good grasp")
            target_z = finger_max_depth
        else:

            if crit>gripper_depth/10*8:
                target_z = palm_max_depth + margin - gripper_depth/10*8
            else:
                target_z = max(palm_max_depth + margin -0.06, finger_max_depth + margin)
        
        world_points = world_to_grasp[bbox_num].copy()
        world_points[2] = target_z


    # plt.figure(figsize=(10,10))
    # plt.scatter(PCD[0],PCD[1], c = PCD[2], cmap = "jet", s=0.1)
    # plt.scatter(world_to_grasp.T[0],world_to_grasp.T[1], c = 'yellow', s=10)
    # pt = gripper_bbox_3d[bbox_num]
    # plt.plot(pt[0,[0,1,2,3,0]],pt[1,[0,1,2,3,0]],c = "g")
    # plt.plot(pt[0,[4,5,6,7,4]],pt[1,[4,5,6,7,4]], c = "g")
    # plt.plot(pt[0,[8,9,10,11,8]],pt[1,[8,9,10,11,8]], c = "r")
    # plt.scatter(pt[0,12],pt[1,12], c = "b",s=5)

    # pt_in_bbx = point_in_bboxes_idx[bbox_num]
    # plt.scatter(world_to_obj_part[0][pt_in_bbx[0]], world_to_obj_part[1][pt_in_bbx[0]], c = "g",s=2)
    # plt.scatter(world_to_obj_part[0][pt_in_bbx[1]], world_to_obj_part[1][pt_in_bbx[1]], c = "g",s=2)
    # plt.scatter(world_to_obj_part[0][pt_in_bbx[2]], world_to_obj_part[1][pt_in_bbx[2]], c = "r",s=2)
    # plt.axis('equal')
    # plt.show()
    return {
        "width" : world_width,
        "world_points" : world_points,
        "angle" : angle,
    }


