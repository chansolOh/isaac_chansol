from client import TcpClient
import numpy as np




if __name__ == "__main__":
    c = TcpClient("192.168.0.137", 9111, name="chansol")
    c.connect()
    joint_position = c.send("get_ik", 
            {
                "target_pos": [0.0, 0.0, 0.3],
                "target_ori": [0, 0, 0],
                "return_traj": True
            }
        )
    joint_position = np.array(joint_position["joint_positions"])
    print(joint_position)



    c.close()
