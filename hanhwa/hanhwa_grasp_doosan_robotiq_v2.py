import rclpy
rclpy.init()
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from socket_utils.client import TcpClient
from camera.get_kinect_img import K4ACamera

from Utils.general_utils.grasp_sample_gen import InteractiveGraspRect
from Utils.general_utils.cal_depth_width import cal_grasp_width_z
from doosan_cont_traj import Controller
from robotiq_gripper import RobotiqGripper_Chansol
from Utils.general_utils import mat_utils

import threading

import matplotlib.pyplot as plt 
import numpy as np
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk
import os 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Toplevel
import json
import cv2
import math

from clients.client_segmentation import DetrClient
from clients.client_grasp_detection import HRDemoGraspPredClient, quat_mul

def is_push(result_arr, key, centroid, r_push=30):
    # ROI 사각형으로 먼저 자르기 (범위 클램프)
    x0 = max(0, centroid[0] - r_push); x1 = min(640, centroid[0] + r_push + 1)
    y0 = max(0, centroid[1] - r_push); y1 = min(480, centroid[1] + r_push + 1)

    roi = result_arr[y0:y1, x0:x1]

    # ROI 내부에서 원 마스크 만들기
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - centroid[0]) ** 2 + (yy - centroid[1]) ** 2 <= r_push * r_push

    vals = roi[mask]
    bObject_detected = np.any((vals != 0) & (vals != key))


    return bObject_detected


def sample_free_centroid(result_arr, centroid, r_search=100, r_check=30, max_trials=200, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    H, W = result_arr.shape[:2]
    cx, cy = int(centroid[0]), int(centroid[1])

    for _ in range(max_trials):
        # --- r_search 원 안에서 무작위 점 샘플링(면적 균일) ---
        theta = rng.uniform(0.0, 2.0 * np.pi)
        rad = r_search * np.sqrt(rng.uniform(0.0, 1.0))  # area-uniform
        x = int(round(cx + rad * np.cos(theta)))
        y = int(round(cy + rad * np.sin(theta)))

        # 이미지 범위 체크
        if not (0 <= x < W and 0 <= y < H):
            continue
        # "다른 centroid" 조건 (완전 동일점 제외)
        if x == cx and y == cy:
            continue

        # --- (x,y) 주변 r_check 원 안에 물체가 있는지 검사 ---
        x0 = max(0, x - r_check); x1 = min(W, x + r_check + 1)
        y0 = max(0, y - r_check); y1 = min(H, y + r_check + 1)

        roi = result_arr[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r_check * r_check

        vals = roi[mask]
        has_obj = np.any(vals != 0)

        if not has_obj:
            return True, [x, y]  # 성공

    return False, None  # 실패(시도 횟수 내에 못 찾음)


def fill_depth_zero_linear(depth: np.ndarray):
    """
    depth: (H,W) np.uint16 or np.float32
    return: depth with 0-valued pixels linearly interpolated
            using valid neighbors (up, down, left, right)
    """
    d = depth.astype(np.float32, copy=True)

    hole = (d == 0)
    if not hole.any():
        return depth

    # 이웃 픽셀 (패딩해서 경계 처리)
    up    = np.pad(d[:-1, :], ((1,0),(0,0)), constant_values=0)
    down  = np.pad(d[1:, :],  ((0,1),(0,0)), constant_values=0)
    left  = np.pad(d[:, :-1], ((0,0),(1,0)), constant_values=0)
    right = np.pad(d[:, 1:],  ((0,0),(0,1)), constant_values=0)

    neighbors = np.stack([up, down, left, right], axis=0)

    valid = neighbors > 0
    count = valid.sum(axis=0)

    # 평균 계산 (0으로 나눔 방지)
    avg = np.sum(neighbors * valid, axis=0) / np.maximum(count, 1)

    # hole 중에서 유효 이웃이 있는 픽셀만 채움
    d[hole & (count > 0)] = avg[hole & (count > 0)]

    return d.astype(depth.dtype)

os.environ["ROS_DOMAIN_ID"] = "19"
os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"

# def center_crop(img, crop_w, crop_h):
#     h, w = img.shape[:2]
#     cx, cy = w // 2, h // 2

#     x1 = max(cx - crop_w // 2, 0)
#     y1 = max(cy - crop_h // 2, 0)
#     x2 = min(cx + crop_w // 2, w)
#     y2 = min(cy + crop_h // 2, h)

#     return img[y1:y2, x1:x2]

def center_crop(img, crop_w, crop_h, offset_w=0, offset_h=0):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    x1 = max(cx - crop_w // 2, 0) + offset_w
    y1 = max(cy - crop_h // 2, 0) + offset_h
    x2 = min(cx + crop_w // 2, w) + offset_w
    y2 = min(cy + crop_h // 2, h) + offset_h

    return img[y1:y2, x1:x2]

class DataGenUI:
    def __init__(self, root):
        self.root = root
        self.root.title("데이터 생성 GUI")
        self.root.geometry("1000x800")
        for i in range(5):  # 예: 총 5열이면
            self.root.grid_columnconfigure(i, weight=1, uniform="col")
        # ===== 입력 필드 =====
        tk.Label(root, text="target_object").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.target_object_entry = tk.Entry(root, width=30)
        self.target_object_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(root, text="gripper_model").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.gripper_model_entry = tk.Entry(root, width=30)
        self.gripper_model_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(root, text="gripper_type").grid(row=3, column=0, padx=10, pady=5, sticky="e")
        self.gripper_type_entry = tk.Entry(root, width=30)
        self.gripper_type_entry.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(root, text="save_path").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.save_path_entry = tk.Entry(root, width=25)
        self.save_path_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        tk.Label(root, text="scene_num").grid(row=4, column=0, padx=10, pady=5, sticky="e")
        self.scene_num_entry = tk.Entry(root, width=25)
        self.scene_num_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        tk.Button(root, text="찾기", command=self.browse_path).grid(row=3, column=2, padx=5, pady=5)

        # ===== 버튼 =====
        tk.Button(root, text="시작", width=10, command=self.start).grid(            row=5, column=0, pady=10 , padx=0)
        tk.Button(root, text="정지", width=10, command=self.stop).grid(             row=5, column=1, pady=10 , padx=0)
        tk.Button(root, text="로봇 reset", width=10, command=self.robot_reset).grid(row=5, column=2, pady=10 , padx=0)
        tk.Button(root, text="그리퍼 open", width=10, command=self.gripper_open).grid(row=5, column=3, pady=10, padx=0)

        tk.Button(root, text="Scene 생성", width=10, command=self.create_scene).grid(row=6, column=0, pady=10)
        tk.Button(root, text="추가", width=10, command=self.add).grid(row=6, column=1, pady=10)
        tk.Button(root, text="저장", width=10, command=self.save).grid(row=6, column=2, pady=10)
        tk.Button(root, text="배치", width=10, command=self.batch).grid(row=7, column=0, pady=10)
        

        self.root.bind("<Escape>", lambda e: self.robot_stop())
        self.scene_img = np.zeros((1080,1920,3), dtype=np.uint8)


        fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.imshow(self.scene_img)
        self.ax.plot([0, 1, 2, 3], [10, 5, 8, 6])
        self.ax.set_title("Sample Plot")

        # ===== Figure를 Tkinter에 붙이기 =====
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=8, column=0, columnspan=3, pady=10)

        self.target_object_entry.insert(0, "obj_000")
        self.gripper_model_entry.insert(0, "UON_Robotics_Jamin_Gripper")
        self.gripper_type_entry.insert(0, "finger2")
        self.save_path_entry.insert(0, "/nas/Dataset/Dataset_2025/dataset_v1/Logistic_site/UONRobitcs_1F/demo1")
        self.path_pred_result = "/home/uon/ochansol/isaac_chansol/hanhwa/pred_results"
        self.grasp_list = []
        self.loop_flag = False
        self.stage = 0


        self.kinect = K4ACamera(resolution="1080P", wb_kelvin=4500, exposure_us=8000)  # 여기만 바꾸면 됨
        self.kinect.start()

        self.cont = Controller()
        self.gripper_cont = RobotiqGripper_Chansol()
        self.gripper_cont.activate()
        self.gripper_cont.calibrate(0, 140)

        self.ik_client = TcpClient("127.0.0.1", 9111, name="chansol")
        self.ik_client.connect()

        self.detr = DetrClient()
        self.node_grasp = HRDemoGraspPredClient()

        self.start_thread()


    def robot_reset(self, sync_type=0):
        self.stop()
        self.cont.reset_robot(sync_type=sync_type)

    def robot_stop(self):
        self.cont.stop()
        self.stage = 6

    def gripper_open(self):
        # self.stop()
        # self.cont.gripper.open()
        pass

    def browse_path(self):
        path = filedialog.askdirectory(title="저장할 폴더 선택")
        if path:
            self.save_path_entry.delete(0, tk.END)
            self.save_path_entry.insert(0, path)

    # ===== 버튼 함수 (사용자가 작성) =====
    def create_scene(self):
        scene_num = int(self.scene_num_entry.get())
        save_path = self.save_path_entry.get()
        if os.path.exists(os.path.join(save_path,"rgb",f"{scene_num:04d}.png")):
            self.scene_img = np.array(Image.open(os.path.join(save_path,"rgb",f"{scene_num:04d}.png")))
            with open(os.path.join(save_path,"output_grasp",f"{scene_num:04d}.json"),'r') as f:
                self.grasp_list = json.load(f)
            self.ax.clear()
            self.ax.imshow(self.scene_img)
            for grasp in self.grasp_list:
                grasp_success = grasp["grasp_success"]

                bbox = np.array(grasp["bbox_2d"]["bbox"])
                self.ax.plot(bbox[[0,1],0],bbox[[0,1],1], color="blue" if grasp_success else "red")
                self.ax.plot(bbox[[2,3],0],bbox[[2,3],1], color="blue" if grasp_success else "red")
                self.ax.plot(bbox[[1,2],0],bbox[[1,2],1], color="green" if grasp_success else "pink")
                self.ax.plot(bbox[[3,0],0],bbox[[3,0],1], color="green" if grasp_success else "pink")
                self.ax.set_title("Updated Plot")

            self.canvas.draw()

            
        else:
            self.scene_img, depth_img = self.kinect.get_rgb_depth()
            Image.fromarray(self.scene_img).save(
                os.path.join(self.save_path_entry.get(),"rgb",f"{int(self.scene_num_entry.get()):04d}.png")
                )
            np.save(os.path.join(self.save_path_entry.get(),"depth",f"{int(self.scene_num_entry.get()):04d}.npy"),depth_img)
            self.grasp_list = []
            self.ax.clear()
            self.ax.imshow(self.scene_img)
            self.canvas.draw()

    def start_thread(self):
        # 별도 스레드에서 start() 실행
        t = threading.Thread(target=self.grasp, daemon=True)
        t.start()

    def grasp(self):
        while True:
            while self.loop_flag:
                ###### init ######

                self.grasp_success = False


                #################### Kinect #################

                self.cont.reset_robot()
                self.gripper_cont.open()

                rgb_img, depth_img = self.kinect.get_rgb_depth()
                depth_img = depth_img.astype(np.float32)
                depth_img = fill_depth_zero_linear(depth_img)
                depth_img = fill_depth_zero_linear(depth_img)
                depth_img = fill_depth_zero_linear(depth_img)
                depth_img = np.where(depth_img==0,1.000, depth_img)
                depth_img = np.clip(depth_img, 0,1.2)
                # plt.imshow(depth_img)
                # plt.show()

                rgb_img_crop = center_crop(rgb_img, 640,480, 0, 250)
                depth_img_crop = center_crop(depth_img,640,480, 0 ,250)
                
                self.ax.clear()
                self.ax.imshow(rgb_img_crop)
                self.canvas.draw()
                #############################################3
                

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                os.makedirs(os.path.join(self.path_pred_result, timestamp), exist_ok=True)

                ################## detr ################3
                result_path = self.detr.send(rgb_img_crop, timestamp)

                if result_path == "Empty":
                    print("No object detected")
                    self.stop()
                    break

                # npy_path = [os.path.join(self.detr_root_path,result_path,i) for i in os.listdir(os.path.join(self.detr_root_path,result_path)) if i.endswith(".npy")][0]
                # json_path = [os.path.join(self.detr_root_path,result_path,i) for i in os.listdir(os.path.join(self.detr_root_path,result_path)) if i.endswith(".json")][0]
                npy_path = os.path.join(self.path_pred_result,timestamp,"segmentation_instance_mask.npy")
                json_path = os.path.join(self.path_pred_result,timestamp,f"segmentation_info.json")


                if not os.path.exists(npy_path):
                    print("################################## npy exist error")
                    time.sleep(1)
                if not os.path.exists(json_path):
                    print("################################## json exist error")
                    time.sleep(1)
                result_arr= np.load(npy_path) # (480,640) instance_id map

                with open(json_path, "r",encoding="utf-8") as f:
                    json_data = json.load(f)

                class_dict = {}
                for data in json_data["objects"].values():
                    if data["class"] == 0:
                        class_dict[data["instance_id"]]= {"class_name":"box"}
                    elif data["class"] == 1:
                        class_dict[data["instance_id"]]= {"class_name":"profile"}
                    class_dict[data["instance_id"]]["bbox"] =data["bbox"]

                depth_mean_best = 2
                depth_top_key = None
                for key in class_dict:
                    bbox = np.array(class_dict[key]["bbox"])
                    if bbox[2]-bbox[0] >= 250 and bbox[3]-bbox[1] >=250:
                        continue
                    
                    mask_y,mask_x = np.where(result_arr==key)
                    depth_mean = depth_img_crop[mask_y,mask_x].min()
 
                    if depth_mean_best > depth_mean :
                        depth_mean_best = depth_mean
                        depth_top_key = key
                
                import random
                depth_top_key = random.choice(list(class_dict.keys()))

                mask_y,mask_x = np.where(result_arr==depth_top_key)
  
                # mask_y,mask_x = np.where(result_arr ==np.unique(result_arr)[2])
                mask_send = np.zeros_like(result_arr).astype(np.uint8)
                mask_send[mask_y,mask_x] = 255
                ###############################################


                ######################### Grasp pred #####################
                # send
                crop_compen_xy = np.array([640,300 + 250])
                # centroid = [int(np.mean(mask_x))  , int(np.mean(mask_y)) ] 

                centroid = [int((mask_x.max()+mask_x.min())/2)   , int((mask_y.max()+mask_y.min())/2) ] 
                print("center : ", centroid)
                cv2.imwrite(f"/home/uon/ochansol/isaac_chansol/hanhwa/pred_results/{timestamp}/mask_depth_top_key({depth_top_key}).png", mask_send)

                centroid, grasp_angle_deg, grasp_width, display_grasp = self.node_grasp.send(gripper_type="ROBOTIS_RH-P12-RN", 
                                                                                   img_rgb=rgb_img_crop[...,:3], 
                                                                                   img_depth=depth_img_crop, 
                                                                                   img_mask=mask_send,
                                                                                   centroids=centroid)
                centroid = [[centroid[i], centroid[i+1]] for i in range(0, len(centroid), 2)]
                classes = []

                idx_delete = []
                for i in range(len(centroid)):
                    key = result_arr[centroid[i][1], centroid[i][0]]
                    if key != 0:
                        classes.append(class_dict[key]["class_name"])
                        idx_delete.append(False)
                    else:
                        idx_delete.append(True)
                        
                print(len(centroid))
                print(len(grasp_angle_deg))
                print(len(grasp_width))
                print("-===")
                print(len(classes))

                idx_delete = np.array(idx_delete, dtype=bool)
                centroid = np.array(centroid)[~idx_delete].tolist()
                grasp_angle_deg = np.array(grasp_angle_deg)[~idx_delete].tolist()
                grasp_width = np.array(grasp_width)[~idx_delete].tolist()
                
                print(len(centroid))
                print(len(grasp_angle_deg))
                print(len(grasp_width))

                class_target = classes[0]
                centroid = centroid[0]
                grasp_angle_deg = grasp_angle_deg[0]
                grasp_width = grasp_width[0]

                # pushing
                r_push = 30
                r_search = 100
                r_check = 30
                if class_target == "box":
                    vis_push = rgb_img_crop[...,:3].copy()
                    bObject_detected = is_push(result_arr, key, centroid, r_push)

                    if bObject_detected:
                        print("object detected (box) : ", bObject_detected)
                        cv2.circle(vis_push, centroid, r_push, (255, 255, 255), 2)
                        plt.imshow(vis_push)
                        plt.show()
                        action_type = "push"
                    else:
                        action_type = "grasp"
                else:
                    action_type = "grasp"



                cv2.imwrite(f"/home/uon/ochansol/isaac_chansol/hanhwa/pred_results/{timestamp}/grasp.png", display_grasp)
                # post processing
                if grasp_width <=30: break
                grasp_angle_rad = np.deg2rad(grasp_angle_deg)
                
                orientation_target = [0., math.sin(grasp_angle_rad / 2.0), 0., math.cos(grasp_angle_rad / 2.0)] # qx, qy, qz, qw
                orientation_base = [0.5, 0.5, 0.5, 0.5]
                orientation = quat_mul(orientation_base, orientation_target) # qx, qy, qz, qw

                print("=====grasp predict results=====")
                print(f"  angle   : {grasp_angle_deg:.2f} deg   ({grasp_angle_rad:.4f} rad)")
                print(f"  width   : {grasp_width} / ???")
                
                height = 30
                if action_type == "push":
                    ####### push end point (centroid) #############
                    x,y = np.array(centroid) + crop_compen_xy
                    width = int(grasp_width*1.2)
                    angle = grasp_angle_deg

                    self.center = [x,y]
                    self.width = width
                    self.height = height
                    self.angle = angle

                    ####### push start point #############
                    success, centroid_push_start = sample_free_centroid(result_arr, centroid, r_search=r_search, r_check=r_check, max_trials=200)
                    if not success:
                        print("No valid push start point found")
                        self.stop()
                        break
                    x_start,y_start = np.array(centroid_push_start) + crop_compen_xy
                    width = int(grasp_width*1.2)
                    angle = grasp_angle_deg

                    self.center = [x_start,y_start]
                    self.width = width
                    self.height = height
                    self.angle = angle

                    dx, dy = centroid[0] - centroid_push_start[0], centroid[1] - centroid_push_start[1]
                    self.ax.arrow(centroid_push_start[0], centroid_push_start[1], dx, dy, head_width=10, head_length=15, length_includes_head=True)
                    self.canvas.draw()

                elif action_type == "grasp":
                    x,y = np.array(centroid) + crop_compen_xy
                    width = int(grasp_width*1.2)
                    angle = grasp_angle_deg

                    self.center = [x,y]
                    self.width = width
                    self.height = height
                    self.angle = angle
                    self.ax.scatter(centroid[0],centroid[1] , s=50, c="r")
                    self.canvas.draw()



                #################### grasp gen ##############
                '''

                fig, ax = plt.subplots(1,2,figsize=(25,18))
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

                for axx in ax.flat:
                    axx.set_xticks([])
                    axx.set_yticks([])
                plt.margins(0, 0)
                plt.gcf().set_tight_layout(False)
                ax[0].imshow(rgb_img)
                ax[1].imshow(depth_img)

                inter = InteractiveGraspRect(ax[0], fixed_height=height)
                plt.show(block=True)

                try:
                    x = int(inter.cx)
                    y = int(inter.cy)
                    angle = -int(inter.angle_deg)
                    width = int(inter.width)
                    width = np.clip(width, 0,140)
                except:
                    self.stop()
                    break
                
                self.center = [x,y]
                self.width = width
                self.height = height
                self.angle = angle
                #####################################################
                '''
                # import pdb;pdb.set_trace()




                self.bbox = [
                    [-width/2, -height/2,0,1],
                    [+width/2, -height/2,0,1],
                    [+width/2, +height/2,0,1],
                    [-width/2, +height/2,0,1]
                ]
                self.bbox = mat_utils.rot_z(-angle).dot(np.array(self.bbox).T)[:2].T + np.array(self.center)
                print(x,y,angle,width)

                #########################




                if action_type == "push":
                    ####### push end point (centroid) ############# 
                    grasp_result_push_centroid = cal_grasp_width_z(x,y,angle,width, depth_img, cam_tf = mat_utils.rot_z(180).dot(mat_utils.trans([-0.015, 0.77-0.007, 1.01] )))
                    # grasp_result_push_centroid = cal_grasp_width_z(x,y,angle,width, depth_img, cam_tf = mat_utils.rot_z(-180).dot(mat_utils.trans(-0.015, 0.77-0.007, 0.85 )))
                    print("world_points :",grasp_result_push_centroid["world_points"])
                    target_position_push_centroid = grasp_result_push_centroid["world_points"]
                    target_angle_push_centroid = grasp_result_push_centroid["angle"]-90

                    ####### push start point #############
                    grasp_result_push = cal_grasp_width_z(x_start,y_start,angle,width, depth_img, cam_tf = mat_utils.rot_z(180).dot(mat_utils.trans([-0.015, 0.77-0.007, 1.01] )))
                    # grasp_result_push = cal_grasp_width_z(x,y,angle,width, depth_img, cam_tf = mat_utils.rot_z(-180).dot(mat_utils.trans(-0.015, 0.77-0.007, 0.85 )))
                    print("world_points :",grasp_result_push["world_points"])
                    target_position_push = grasp_result_push["world_points"]
                    target_angle_push = grasp_result_push["angle"]-90
                elif action_type == "grasp":
                    ####### cal depth width #############
                    grasp_result = cal_grasp_width_z(x,y,angle,width, depth_img, cam_tf = mat_utils.rot_z(180).dot(mat_utils.trans([-0.015, 0.77-0.007, 1.01] )))
                    # grasp_result = cal_grasp_width_z(x,y,angle,width, depth_img, cam_tf = mat_utils.rot_z(-180).dot(mat_utils.trans(-0.015, 0.77-0.007, 0.85 )))
                    print("world_points :",grasp_result["world_points"])
                    target_position = grasp_result["world_points"]
                    self.grasp_world_points = target_position
                    # target_position[-1] = np.clip(target_position[-1], 0, 0.005)
                    
                    target_angle = grasp_result["angle"]-90
                    # import pdb;pdb.set_trace()
                    self.target_width = np.clip(grasp_result["width"],0,0.137)*1000
                else:
                    raise RuntimeError("action type error")


                ####### robot control ###############
                z_via = 0.2
                z_margin = -0.01
                z_clip_th = 0.02
                time_th = 2

                action_flag = True
                self.stage = 0
                while self.loop_flag:

                    if self.cont.emergency_stop_flag:
                        self.stop()
                        self.cont.emergency_stop_flag = False

                    # grasp actions
                    if action_type == "grasp":
                        ##### stage 0   via up ######
                        if self.stage == 0:
                            if action_flag:
                                gripper_tar = self.target_width
                                print("gripper_tar : ", gripper_tar)
                                joint_states = np.array(self.cont.joint_states)
                                target_position[2] += z_via + z_margin
                                target_position[2] = np.clip(target_position[-1], z_clip_th, 0.3)
                                print(f"stage_{self.stage}-target_pos :", target_position)
                                joint_position = self.ik_client.send("get_ik", 
                                    {
                                        "target_pos": target_position.tolist(),
                                        "target_ori": np.array([0,0,target_angle]).tolist(),
                                        "frame_name": "Robotiq_2f140_open",
                                        "init_joint_state": joint_states.tolist(),
                                        "return_traj": False
                                    }
                                )
                                joint_position = np.array(joint_position["joint_positions"])
                                print(f'Joint Position: {joint_position}')

                                self.gripper_cont.goTomm(width,speed=255,force=100, sync=False)

                                self.cont.movej(pos= joint_position,
                                                vel=10.0,
                                                sync_type=1)
                                
                                action_flag = False
                                start_time=time.time()

                            else:
                                dur_time = time.time() - start_time

                                # gripper_st, gripper_obj = self.cont.gripper.read_pos_obj()
                                joint_st = np.array(self.cont.joint_states)
                                if np.sum( ( joint_position/180*np.pi - joint_st)**2)**(1/2) < 2/180*np.pi :
                                    self.stage+=1
                                    action_flag = True
    
                            
                        ##### stage 1   down ######
                        if self.stage == 1:
                            
                            if action_flag:
                                # gripper_tar = self.target_width
                                joint_states = np.array(self.cont.joint_states)

                                target_position[2] -=  (z_via-z_margin) - self.gripper_cont.mmToZhop(gripper_tar)/1000
                                target_position[2] = np.clip(target_position[-1], z_clip_th, 0.25)
                                print(f"stage_{self.stage}-target_pos :", target_position)
                                joint_positions = self.ik_client.send("get_ik", 
                                    {
                                        "target_pos": target_position.tolist(),
                                        "target_ori": np.array([0,0,target_angle]).tolist(),
                                        "frame_name": "Robotiq_2f140_open",
                                        "init_joint_state": joint_states.tolist(),
                                        "return_traj": True
                                    }
                                )
                                joint_positions = np.array(joint_positions["joint_positions"])
                                print(f'Joint Position: {joint_positions}')

                                self.cont.send_joint_trajectory(
                                    points=joint_positions,
                                    times_sec=np.linspace(0.2, len(joint_positions)*0.2, len(joint_positions)),
                                    include_start_from_current=True,   # 첫점 현재값 자동 추가
                                    start_hold_sec=0.1,
                                    wait_result=True,
                                    degrees=True,
                                )

                                action_flag = False
                                start_time=time.time()
                            else:
                                dur_time = time.time() - start_time

                                joint_st = np.array(self.cont.joint_states)
                                if np.sum( ( np.array(joint_positions[-1])/180*np.pi - joint_st)**2)**(1/2) < 0.5/180*np.pi :
                                    self.stage+=1
                                    action_flag = True
                                    self.grasp_joint_state = joint_st

                        ##### stage 2   close gripper ######
                        if self.stage == 2:
                            
                            if action_flag:
                                gripper_tar = 0
                                joint_states = np.array(self.cont.joint_states)
                                target_position[2] += self.gripper_cont.mmToZhop(gripper_tar)/1000
                                target_position[-1] = np.clip(target_position[-1], z_clip_th, 0.3)

                                joint_position = self.ik_client.send("get_ik", 
                                    {
                                        "target_pos": target_position.tolist(),
                                        "target_ori": np.array([0,0,target_angle]).tolist(),
                                        "frame_name": "Robotiq_2f140_open",
                                        "init_joint_state": joint_states.tolist(),
                                        "return_traj": False
                                    }
                                )
                                joint_position = np.array(joint_position["joint_positions"])
                                print(f'Joint Position: {joint_position}')


                                self.gripper_cont.goTomm(gripper_tar,speed=81,force=255, sync=False) ##100mm/s
                                # self.cont.send_joint_trajectory(
                                #     points=joint_positions,
                                #     times_sec=np.linspace(0.2, len(joint_positions)*0.2, len(joint_positions)),
                                #     include_start_from_current=False,   # 첫점 현재값 자동 추가
                                #     start_hold_sec=0,
                                #     wait_result=False,
                                #     degrees=True,
                                # )
                                self.cont.movej(pos= joint_position,
                                                vel=10.0,
                                                sync_type=1)

                                action_flag = False
                                start_time=time.time()
                            else:
                                dur_time = time.time() - start_time
                                gripper_st, gripper_obj = self.gripper_cont.read_pos_obj()
                                joint_st = np.array(self.cont.joint_states)
                                if (np.sum((gripper_tar - gripper_st)**2)**(1/2) <1 and \
                                    np.sum( ( joint_position/180*np.pi - joint_st)**2)**(1/2) < 0.5/180*np.pi) or dur_time>time_th:
                                    self.stage+=1
                                    action_flag = True
                                    self.grasp_joint_state = joint_st

        


                        ##### stage 3   via up ######
                        if self.stage == 3:
                            if action_flag:
                                joint_states = np.array(self.cont.joint_states)
                                target_position[2] += (z_via - z_margin)
                                target_position[2] = np.clip(target_position[-1], z_clip_th, 0.3 )
                                joint_positions = self.ik_client.send("get_ik", 
                                    {
                                        "target_pos": target_position.tolist(),
                                        "target_ori": np.array([0,0,target_angle]).tolist(),
                                        "frame_name": "Robotiq_2f140_open",
                                        "init_joint_state": joint_states.tolist(),
                                        "return_traj": True
                                    }
                                )
                                joint_positions = np.array(joint_positions["joint_positions"])
                                print(f'Joint Position: {joint_positions}')

                                self.cont.send_joint_trajectory(
                                    points=joint_positions,
                                    times_sec=np.linspace(0.2, len(joint_positions)*0.2, len(joint_positions)),
                                    include_start_from_current=True,   # 첫점 현재값 자동 추가
                                    start_hold_sec=0.1,
                                    wait_result=True, # False??
                                    degrees=True,
                                )
                                action_flag = False
                                
                                start_time=time.time()
                            else:
                                dur_time = time.time() - start_time
                                joint_st = np.array(self.cont.joint_states)
                                if np.sum( ( np.array(joint_positions[-1])/180*np.pi - joint_st)**2)**(1/2) < 2/180*np.pi:
                                    self.stage+=1
                                    action_flag = True

                        ##### stage 4   move to place ######
                        if self.stage == 4:
                            if action_flag:
                                joint_states = np.array(self.cont.joint_states)
                                if class_target == "profile":
                                    target_position = np.array([0.15,-0.95,0.25])
                                elif class_target == "box":
                                    target_position = np.array([-0.15,-0.95,0.25])
                                else : 
                                    key_command = input("key input : ")
                                    if key_command == int(0): target_position = np.array([0.2,-0.9,0.06])
                                    else : target_position = np.array([-0.2,-0.9,0.06])

                                joint_position = self.ik_client.send("get_ik", 
                                    {
                                        "target_pos": target_position.tolist(),
                                        "target_ori": np.array([0,0,target_angle]).tolist(),
                                        "frame_name": "Robotiq_2f140_close",
                                        "init_joint_state": joint_states.tolist(),
                                        "return_traj": False
                                    }
                                )
                                joint_position = np.array(joint_position["joint_positions"])
                                print(f'Joint Position: {joint_position}')
                                self.cont.movej(pos= joint_position,
                                                vel=10.0,
                                                sync_type=1)
                                action_flag = False
                                
                                start_time=time.time()
                            else:
                                dur_time = time.time() - start_time
                                joint_st = np.array(self.cont.joint_states)
                                if np.sum( ( joint_position/180*np.pi - joint_st)**2)**(1/2) < 2/180*np.pi  :
                                    self.stage+=1
                                    action_flag = True

                        ##### stage 5      open gripper ######
                        if self.stage == 5:
                            self.gripper_cont.open()
                            # self.add()
                            self.stage += 1

                        ##### stage 6   move home ######
                        if self.stage == 6:
                            if action_flag:
                                joint_states = np.array(self.cont.joint_states)
    

                                target_position = np.array([0,-0.5,0.25])
                                joint_position = self.ik_client.send("get_ik", 
                                    {
                                        "target_pos": target_position.tolist(),
                                        "target_ori": np.array([0,0,target_angle]).tolist(),
                                        "frame_name": "Robotiq_2f140_open",
                                        "init_joint_state": joint_states.tolist(),
                                        "return_traj": False
                                    }
                                )
                                joint_position = np.array(joint_position["joint_positions"])
                                print(f'Joint Position: {joint_position}')
                                self.cont.movej(pos= joint_position,
                                                vel=10.0,
                                                sync_type=1)
                                action_flag = False
                                

                            else:
                                joint_st = np.array(self.cont.joint_states)
                                if np.sum( ( joint_position/180*np.pi - joint_st)**2)**(1/2) < 2/180*np.pi :

                                    self.stage+=1
                                    action_flag = True
                        if self.stage == 7:
                            break
                    
                    if action_type == "push":
                        ##### stage 0   via up ######
                        if self.stage == 0:
                            if action_flag:
                                joint_states = np.array(self.cont.joint_states)
                                target_position_push[2] += z_via + z_margin
                                target_position_push[2] = np.clip(target_position_push[-1], z_clip_th, 0.3)
                                print(f"stage_{self.stage}-target_pos :", target_position_push)
                                joint_position = self.ik_client.send("get_ik", 
                                    {
                                        "target_pos": target_position_push.tolist(),
                                        "target_ori": np.array([0,0,target_angle_push]).tolist(),
                                        "frame_name": "Robotiq_2f140_open",
                                        "init_joint_state": joint_states.tolist(),
                                        "return_traj": False
                                    }
                                )
                                joint_position = np.array(joint_position["joint_positions"])
                                print(f'Joint Position: {joint_position}')

                                self.gripper_cont.goTomm(width,speed=255,force=100, sync=False)

                                self.cont.movej(pos= joint_position,
                                                vel=10.0,
                                                sync_type=1)
                                
                                action_flag = False
                                start_time=time.time()

                            else:
                                dur_time = time.time() - start_time

                                # gripper_st, gripper_obj = self.cont.gripper.read_pos_obj()
                                joint_st = np.array(self.cont.joint_states)
                                if np.sum( ( joint_position/180*np.pi - joint_st)**2)**(1/2) < 2/180*np.pi :
                                    self.stage+=1
                                    action_flag = True
    
                        ##### stage 1   close gripper ######
                        if self.stage == 1:
                            
                            if action_flag:
                                gripper_tar = 0
                                self.gripper_cont.goTomm(gripper_tar,speed=81,force=255, sync=True) ##100mm/s
                                action_flag = False
                            else:
                                self.stage+=1
                                action_flag = True

                            
                        ##### stage 2   down ######
                        if self.stage == 2:
                            
                            if action_flag:
                                # gripper_tar = self.target_width
                                joint_states = np.array(self.cont.joint_states)

                                target_position_push[2] = 0.06
                                print(f"stage_{self.stage}-target_pos :", target_position_push)
                                joint_positions = self.ik_client.send("get_ik", 
                                    {
                                        "target_pos": target_position_push.tolist(),
                                        "target_ori": np.array([0,0,target_angle_push]).tolist(),
                                        "frame_name": "Robotiq_2f140_open",
                                        "init_joint_state": joint_states.tolist(),
                                        "return_traj": True
                                    }
                                )
                                joint_positions = np.array(joint_positions["joint_positions"])
                                print(f'Joint Position: {joint_positions}')

                                self.cont.send_joint_trajectory(
                                    points=joint_positions,
                                    times_sec=np.linspace(0.2, len(joint_positions)*0.2, len(joint_positions)),
                                    include_start_from_current=True,   # 첫점 현재값 자동 추가
                                    start_hold_sec=0.1,
                                    wait_result=True,
                                    degrees=True,
                                )

                                action_flag = False
                                start_time=time.time()
                            else:
                                dur_time = time.time() - start_time

                                joint_st = np.array(self.cont.joint_states)
                                if np.sum( ( np.array(joint_positions[-1])/180*np.pi - joint_st)**2)**(1/2) < 0.5/180*np.pi :
                                    self.stage+=1
                                    action_flag = True
                                    self.grasp_joint_state = joint_st


                        ##### stage 3   push ######
                        if self.stage == 3:
                            if action_flag:
                                joint_states = np.array(self.cont.joint_states)
                                target_position_push_centroid[2] = 0.06
                                joint_positions = self.ik_client.send("get_ik", 
                                    {
                                        "target_pos": target_position_push_centroid.tolist(),
                                        "target_ori": np.array([0,0,target_angle_push_centroid]).tolist(),
                                        "frame_name": "Robotiq_2f140_open",
                                        "init_joint_state": joint_states.tolist(),
                                        "return_traj": True
                                    }
                                )
                                joint_positions = np.array(joint_positions["joint_positions"])
                                print(f'Joint Position: {joint_positions}')

                                self.cont.send_joint_trajectory(
                                    points=joint_positions,
                                    times_sec=np.linspace(0.2, len(joint_positions)*0.2, len(joint_positions)),
                                    include_start_from_current=True,   # 첫점 현재값 자동 추가
                                    start_hold_sec=0.1,
                                    wait_result=True, # False??
                                    degrees=True,
                                )
                                action_flag = False
                                
                                start_time=time.time()
                            else:
                                dur_time = time.time() - start_time
                                joint_st = np.array(self.cont.joint_states)
                                if np.sum( ( np.array(joint_positions[-1])/180*np.pi - joint_st)**2)**(1/2) < 2/180*np.pi:
                                    self.stage+=1
                                    action_flag = True
        


                        ##### stage 4   via up ######
                        if self.stage == 4:
                            if action_flag:
                                joint_states = np.array(self.cont.joint_states)
                                target_position_push_centroid[2] += (z_via - z_margin)
                                target_position_push_centroid[2] = np.clip(target_position_push_centroid[-1], z_clip_th, 0.3 )
                                joint_positions = self.ik_client.send("get_ik", 
                                    {
                                        "target_pos": target_position_push_centroid.tolist(),
                                        "target_ori": np.array([0,0,target_angle_push_centroid]).tolist(),
                                        "frame_name": "Robotiq_2f140_open",
                                        "init_joint_state": joint_states.tolist(),
                                        "return_traj": True
                                    }
                                )
                                joint_positions = np.array(joint_positions["joint_positions"])
                                print(f'Joint Position: {joint_positions}')

                                self.cont.send_joint_trajectory(
                                    points=joint_positions,
                                    times_sec=np.linspace(0.2, len(joint_positions)*0.2, len(joint_positions)),
                                    include_start_from_current=True,   # 첫점 현재값 자동 추가
                                    start_hold_sec=0.1,
                                    wait_result=True, # False??
                                    degrees=True,
                                )
                                action_flag = False
                                
                                start_time=time.time()
                            else:
                                dur_time = time.time() - start_time
                                joint_st = np.array(self.cont.joint_states)
                                if np.sum( ( np.array(joint_positions[-1])/180*np.pi - joint_st)**2)**(1/2) < 2/180*np.pi:
                                    self.stage+=1
                                    action_flag = True

                        ##### stage 5      open gripper ######
                        if self.stage == 5:
                            self.gripper_cont.open()
                            # self.add()
                            self.stage += 1

                        ##### stage 6   move home ######
                        if self.stage == 6:
                            if action_flag:
                                joint_states = np.array(self.cont.joint_states)
    

                                target_position = np.array([0,-0.5,0.25])
                                joint_position = self.ik_client.send("get_ik", 
                                    {
                                        "target_pos": target_position.tolist(),
                                        "target_ori": np.array([0,0,target_angle_push_centroid]).tolist(),
                                        "frame_name": "Robotiq_2f140_open",
                                        "init_joint_state": joint_states.tolist(),
                                        "return_traj": False
                                    }
                                )
                                joint_position = np.array(joint_position["joint_positions"])
                                print(f'Joint Position: {joint_position}')
                                self.cont.movej(pos= joint_position,
                                                vel=10.0,
                                                sync_type=1)
                                action_flag = False
                                

                            else:
                                joint_st = np.array(self.cont.joint_states)
                                if np.sum( ( joint_position/180*np.pi - joint_st)**2)**(1/2) < 2/180*np.pi :

                                    self.stage+=1
                                    action_flag = True
                        if self.stage == 7:
                            break

                self.stop()
                # self.save()
 
                            


    def start(self):
        self.loop_flag = True
        print("start : ",self.loop_flag)
    def stop(self):
        self.loop_flag = False
        print("start : ",self.loop_flag)

    def add(self):
        self.grasp_list.append({
            "bbox_2d": {
                "bbox": self.bbox.astype(np.int).tolist(),
                "center": self.center,
                "width": int(self.width),
                "height": int(self.height),
                "angle": self.angle
            },
            "target_points": self.grasp_world_points.round(5).tolist(),
            "target_orientation": [0,0,self.angle],
            "target_width": round(self.target_width/1000,6),
            "target_object": self.target_object_entry.get(),
            "gripper_model": self.gripper_model_entry.get(),
            "gripper_type": self.gripper_type_entry.get(),
            "disturbed_object_count": 1,
            "close_width" : round(self.gripper_close_width/1000,6),
            "target_joint_state" : self.grasp_joint_state.round(5).tolist(),
            "grasp_success" : self.grasp_success
        })
        self.ax.plot(self.bbox[[0,1],0],self.bbox[[0,1],1], color="blue" if self.grasp_success else "red")
        self.ax.plot(self.bbox[[2,3],0],self.bbox[[2,3],1], color="blue" if self.grasp_success else "red")
        self.ax.plot(self.bbox[[1,2],0],self.bbox[[1,2],1], color="green" if self.grasp_success else "pink")
        self.ax.plot(self.bbox[[3,0],0],self.bbox[[3,0],1], color="green" if self.grasp_success else "pink")
        self.ax.set_title("Updated Plot")

        # 다시 렌더링
        self.canvas.draw()

    def save(self):
        pass
        # # import pdb;pdb.set_trace()
        # with open(os.path.join(self.save_path_entry.get(),"output_grasp",f"{int(self.scene_num_entry.get()):04d}.json"),'w') as f:
        #     json.dump(self.grasp_list, f,indent=4)

    
    def batch(self):
        # self.gripper_open()
        self.robot_reset(sync_type=1)
        OverlayWindow(self.root, self.scene_img, self.kinect)


class OverlayWindow:
    def __init__(self, root, base_img,ros_node):
        self.root = root
        self.base_img = Image.fromarray(base_img).convert("RGB")  # PIL 이미지
        self.ros_node = ros_node
        self.alpha = 0.5  # 초기 투명도

        self.top = Toplevel(root)
        self.top.title("Overlay Viewer")

        # 이미지 표시용 Label
        self.label = tk.Label(self.top)
        self.label.pack()

        # 투명도 조절 슬라이더
        self.slider = tk.Scale(
            self.top, from_=0, to=100, orient="horizontal",
            label="Base Image Transparency (%)",
            command=self.update_alpha
        )
        self.slider.set(int(self.alpha * 100))
        self.slider.pack(fill="x", padx=10, pady=10)

        self.update_frame()

    def update_alpha(self, val):
        self.alpha = float(val) / 100.0

    def update_frame(self):
        frame_img = Image.fromarray(self.ros_node.get_rgb()).convert("RGB")
        # 투명도 적용 (base_img: 고정 이미지, frame_img: 실시간)
        w, h = frame_img.size
        w =int(w*0.7)
        h =int(h*0.7)
        blended = Image.blend(frame_img.resize((w,h)), self.base_img.resize((w,h)), self.alpha)

        self.tk_img = ImageTk.PhotoImage(blended)
        self.label.config(image=self.tk_img)

        self.top.after(33, self.update_frame)

    def close(self):
        self.top.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataGenUI(root)
    root.mainloop()
    




