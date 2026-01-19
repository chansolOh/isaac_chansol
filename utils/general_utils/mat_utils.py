import numpy as np
# from PIL import Image
import math


def trans(arr):
    x, y, z = arr
    return np.array([[1,0,0,x],
                     [0,1,0,y],
                     [0,0,1,z],
                     [0,0,0,1]])
def rotate(arr, order="xyz"):
    deg1, deg2, deg3 = arr
    rot_dict = {
        "x": rot_x,
        "y": rot_y,
        "z": rot_z,
        "X": rot_x,
        "Y": rot_y,
        "Z": rot_z,
    }
    order = list(order)
    return mat_dot([ rot_dict[order[-1]](deg3), rot_dict[order[-2]](deg2), rot_dict[order[-3]](deg1)])

def transform(position, rotation, order="xyz"):
    x,y,z = position
    roll,pitch,yaw = rotation
    
    rotate_mat = rotate(roll,pitch,yaw, order=order)
    trans_mat = trans(x,y,z)
    return np.dot(trans_mat, rotate_mat)

def rot_x(deg):
    deg = deg/180*np.pi
    return np.array([[1,0,0,0],
                     [0,np.cos(deg),-np.sin(deg),0],
                     [0,np.sin(deg),np.cos(deg),0],
                     [0,0,0,1]])
    
def rot_y(deg):
    deg = deg/180*np.pi
    return np.array([[np.cos(deg),0,np.sin(deg),0],
                     [0,1,0,0],
                     [-np.sin(deg),0,np.cos(deg),0],
                     [0,0,0,1]])

def rot_z(deg):
    deg = deg/180*np.pi
    return np.array([[np.cos(deg),-np.sin(deg),0,0],
                     [np.sin(deg),np.cos(deg),0,0],
                     [0,0,1,0],
                     [0,0,0,1]])

def mat_dot(matrix_list):
    result = np.eye(4)
    for mat in matrix_list:
        if type(mat) != np.ndarray:
            mat = np.array(mat)
        if mat.shape == (2,2) or mat.shape == (3,3):
            mat = mat_to_tf(mat)
        result = np.dot(result, mat)
    return result


def mat_to_tf(mat):
    if type(mat) == list:
        mat = np.array(mat)
    elif type(mat) != np.ndarray:
        import torch
        eye = torch.eye(4, device=mat.device)
        row,col = mat.shape
        row_mat = eye[row:,:col]
        col_mat = eye[:,col:]
        return torch.concat((torch.concat((mat,row_mat),dim=0),col_mat),dim=1)
    else :
        eye = np.eye(4)
        row,col = mat.shape
        row_mat = eye[row:,:col]
        col_mat = eye[:,col:]
        return np.hstack((np.vstack((mat,row_mat)),col_mat))





def quat_to_euler(arr, degrees=True):
    w, x, y, z = arr
    """
    Quaternion -> Euler (XYZ, roll-pitch-yaw)
    q: [x, y, z, w]
    return: roll, pitch, yaw (rad)
    """
    # roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (Y)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    if degrees:
        roll = roll / np.pi * 180.0
        pitch = pitch / np.pi * 180.0
        yaw = yaw / np.pi * 180.0
    return np.array([roll, pitch, yaw])


def euler_to_quat(arr, degrees=True):
    roll, pitch, yaw = arr
    """
    Euler (XYZ, roll-pitch-yaw) -> Quaternion
    """
    if degrees:
        roll = roll / 180.0 * np.pi
        pitch = pitch / 180.0 * np.pi
        yaw = yaw / 180.0 * np.pi
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])



def mat_to_euler(T, degrees=True):
    """
    T: (4,4) homogeneous transform or (3,3) rotation matrix
    """
    T = np.asarray(T, dtype=float)
    if T.shape == (4, 4):
        R = T[:3, :3]
        t = T[:3, 3].copy()
    elif T.shape == (3, 3):
        R = T
        t = None
    else:
        raise ValueError("T must be shape (4,4) or (3,3)")


    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-9

    if not singular:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock 근처
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0

    if degrees:
        roll = roll / np.pi * 180.0
        pitch = pitch / np.pi * 180.0
        yaw = yaw / np.pi * 180.0

    return np.array([roll, pitch, yaw])




def quat_to_axis_angle(quat):

    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den



def axis_angle_to_quat(axis_angle):
    angle = np.linalg.norm(axis_angle)

    # no rotation
    if angle < 1e-9:
        return np.array([0, 0, 0, 1])   # identity quaternion

    axis = axis_angle / angle
    half = angle / 2.0

    q_xyz = axis * np.sin(half)
    q_w = np.cos(half)

    return np.concatenate([q_xyz, [q_w]])


def quat_mul(q1, q2):
    # (x,y,z,w) format
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 + y1*w2 + z1*x2 - x1*z2,
        w1*z2 + z1*w2 + x1*y2 - y1*x2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])




def dist_based_sampling(points, dist_th):
    ##### points : (2,N) image index
    idx_dist = np.tile(points[:,None,:],(1,len(points.T),1)) - np.tile(points[...,None],(1,1,len(points.T)))
    idx_dist = np.sqrt(np.sum(idx_dist**2,axis=0))

    cnt = 0
    while len(idx_dist)>cnt:
        min_idx = np.argwhere((idx_dist[cnt]<=dist_th) & (idx_dist[cnt]!=0))
        points = np.delete(points, min_idx, axis=1)
        idx_dist = np.delete(idx_dist, min_idx, axis=0)
        idx_dist = np.delete(idx_dist, min_idx, axis=1)
        cnt+=1
    return points




############     specific task set  #############################
def align_bbox(bbox):
    y_min = np.min(bbox[:,1],axis=0)
    # y_min_idx = np.argwhere(bbox[:,1]==y_min)
    y_min_idx = np.where(bbox[:,1]==y_min)[0]
    if y_min_idx.shape[0]>=2:
        x_max_idx = np.argmax(bbox[y_min_idx].T[0])

        return np.roll(bbox, -y_min_idx[x_max_idx],axis=0)
    else:
        return np.roll(bbox,-y_min_idx[0],axis=0)

def lin_func(bbox,points):# bbox=2x2, points = 2xN
    x1,y1 = bbox[0]
    x2,y2 = bbox[1]
    if x2-x1 == 0:
        return points[0] - x1
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1    
    return a*points[0] - points[1] + b 

def select_point_in_bbox(bbox,points,debug=False): # bbox = 4x2, points = 2xN
    # if debug:
    #     import pdb;pdb.set_trace()
    aligned_bbox = align_bbox(bbox)
    p1 = np.where(lin_func(aligned_bbox[[3,2]],points)>=0)
    p2 = np.where(lin_func(aligned_bbox[[2,1]],points)>=0)
    p3 = np.where(lin_func(aligned_bbox[[1,0]],points)<=0)
    p4 = np.where(lin_func(aligned_bbox[[0,3]],points)<=0)

    filtered_idx = np.intersect1d(np.intersect1d(p1,p2),np.intersect1d(p3,p4))
    return filtered_idx

def select_points(bboxes,pt,debug=False) : # bboxes = bboxes x 3 x 13, points = 3 x N
    points = pt.copy()
    idx = []
    for bbox in bboxes:
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox[0].min(), bbox[1].min(), bbox[0].max(), bbox[1].max()
        pts_idx = np.where((points[0]>=bbox_xmin) & (points[0]<=bbox_xmax) & (points[1]>=bbox_ymin) & (points[1]<=bbox_ymax))[0]
        # if debug:
        #     import pdb;pdb.set_trace()
        tmp = [] # 3x2xN
        for i in range(bbox.shape[1]//4):
            tmp.append(pts_idx[select_point_in_bbox(bbox[:2, i*4:(i+1)*4].T, points[:2, pts_idx] ,debug=debug)])
        # tmp.append(pts_idx[select_point_in_bbox(bbox[:2,  :4].T, points[:2,pts_idx])])
        # tmp.append(pts_idx[select_point_in_bbox(bbox[:2, 4:8].T, points[:2,pts_idx])])
        # tmp.append(pts_idx[select_point_in_bbox(bbox[:2, 8:12].T,points[:2,pts_idx])])
        idx.append(tmp)

    return idx #boxes x 3 x 2 x N

###############################





