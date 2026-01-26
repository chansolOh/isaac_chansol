from pyrobotiqgripper import RobotiqGripper
import time
import numpy as np
import json
class RobotiqGripper_Chansol(RobotiqGripper):
    def __init__(self, portname='auto', slaveAddress=9):
        super().__init__(portname, slaveAddress)
        self.open_zhop = 0
        self.close_zhop = 12
    
    def goTo(self,position,speed=255,force=255, sync = True):
        """Go to the position with determined speed and force.
        
        Args:
            - position (int): Position of the gripper. Integer between 0 and 255.\
            0 being the open position and 255 being the close position.
            - speed (int): Gripper speed between 0 and 255
            - force (int): Gripper force between 0 and 255
        
        Returns:
            - objectDetected (bool): True if object detected
            - position (int): End position of the gripper in bits
        """
        position=int(position)
        speed=int(speed)
        force=int(force)
        
        
        #Check if the grippre is activated
        if self.isActivated == False:
            raise Exception ("Gripper must be activated before requesting\
                             an action.")

        #Check input value
        if position>255:
            raise Exception("Position value cannot exceed 255")
        elif position<0:
            raise Exception("Position value cannot be under 0")
        
        self.processing=True
        

        #rARD(5) rATR(4) rGTO(3) rACT(0)
        #gACT=1 (Gripper activation.) and gGTO=1 (Go to Position Request.)
        self.write_registers(1000,[0b0000100100000000,
                                    position,
                                    speed * 0b100000000 + force])
        
        # Waiting for activation to complete
        if sync:
            motionStartTime=time.time()
            motionCompleted=False
            motionTime=0
            objectDetected=False

            while (not objectDetected) and (not motionCompleted)\
                and (motionTime<self.timeOut):

                motionTime= time.time()- motionStartTime
                self.readAll()
                #Object detection status, is a built-in feature that provides
                #information on possible object pick-up. Ignore if gGTO == 0.
                gOBJ=self.paramDic["gOBJ"]

                
                if gOBJ==1 or gOBJ==2: 
                    #Fingers have stopped due to a contact
                    objectDetected=True
                
                elif gOBJ==3:
                    #Fingers are at requested position.
                    objectDetected=False
                    motionCompleted=True
            
            if motionTime>self.timeOut:
                raise Exception("Gripper never reach its requested position and\
                                no object have been detected")
            
            position=self.paramDic["gPO"]

            return position, objectDetected
        

    def goTomm(self,positionmm,speed=255,force=255, sync = True):
        """Go to the requested opening expressed in mm

        Args:
            - positionmm (float): Gripper opening in mm.
            - speed (int, optional): Gripper speed between 0 and 255.\
            Default is 255.
            - force (int, optional): Gripper force between 0 and 255.\
            Default is 255.
        
        .. note::
            Calibration is needed to use this function.\n
            Execute the function calibrate at least 1 time before using this function.
        """
        if self.isCalibrated == False:
            raise Exception("The gripper must be calibrated before been requested to go to a position in mm")

        if  positionmm>self.openmm:
            raise Exception("The maximum opening is {}".format(self.openmm))
        
        position=int(self._mmToBit(positionmm))
        # position=int(self.mmToBit_chansol(positionmm))
        self.goTo(position,speed,force, sync=sync)
    
    def read_pos_obj(self):
        pos = self.getPositionmm()
        obj = self.paramDic["gOBJ"]
        return pos,obj

    #deprecated
    def mmToZhop_exp(self, mm):
        aCoef_z=(self.close_zhop-self.open_zhop)/(self.closebit-self.openbit)
        bCoef_z=(self.open_zhop*self.closebit-self.openbit*self.close_zhop)/(self.closebit-self.openbit)
        
        bit=(mm-self._bCoef)/self._aCoef
        z_hop=aCoef_z*bit+bCoef_z
        return z_hop
    def mmToZhop(self, mm):
        # Load calibration parameters
        with open("/home/uon/ochansol/isaac_chansol/hanhwa/robotiq_zhop_width_rate.json", "r") as f:
            json_data = json.load(f)
        a =  json_data["params"]["a"]
        b =  json_data["params"]["b"]
        c =  json_data["params"]["c"]
        d =  json_data["params"]["d"]
        json_data["data"]
        # Calculate bit value from mm using the cubic polynomial
        z_hop = a * (mm ** 3) + b * (mm ** 2) + c * mm + d
        return round(25.2-z_hop,1)
    
    def mmToZhop_seq(self, mm):
        mm_array = np.arange(mm,0,-10)
        zhop_array = []
        for mm in mm_array:
            zhop = self.mmToZhop(mm)
            zhop_array.append(zhop)
        return zhop_array
    
    def open(self,speed=255,force=255):
        self.goTo(0,force,speed)

if __name__ == "__main__":
    gripper_cont = RobotiqGripper_Chansol()
    gripper_cont.activate()
    gripper_cont.calibrate(0, 140)

    # gripper_cont.goTomm(140)
    import pdb; pdb.set_trace()