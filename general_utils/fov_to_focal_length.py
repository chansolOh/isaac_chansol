import numpy as np


def fov_to_focal_length(fov, pixel_size, sensor_size):
    fx = sensor_size[0]/(2*np.tan(np.deg2rad(fov[0]/2)))
    fy = sensor_size[1]/(2*np.tan(np.deg2rad(fov[1]/2)))
    return {
        "intrinsic_matrix": [[fx,0,sensor_size[0]/2],[0,fy,sensor_size[1]/2],[0,0,1]],
        "isaac_focal_length": round((fx+fy)/2*pixel_size,5),
        "horizontal_aperture": round( sensor_size[0]*pixel_size,5)
    }
    
 

if __name__ == "__main__":
    # fov = [80,51]
    # fov = [75,65] #depth camera
    fov = [90,59] #degrees
    pixel_size = 0.00125 #mm
    sensor_size = [[3840,2160],
                [2560,1440],
                [1920,1080],
                [1280,720],
                [640,576],] #pixels
    for size in sensor_size:
        print(fov_to_focal_length(fov, pixel_size, size))