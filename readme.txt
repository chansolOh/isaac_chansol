# how to run
1. 카메라 렉 덜걸림 
$ echo 16000 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb

2. inverse server(isaac)
$ source ~/ochansol/isaac_chansol/.venv/bin/activate
$ python ~/ochansol/isaac_chansol/socket_utils/isaac_ikserver.py

3. doosan
$ ros2 launch dsr_bringup2 dsr_bringup2_moveit.launch.py mode:=real host:=192.168.100.101 model:=m1013

4. detector
$ source ~/Documents/Detector_Demo/env/bin/activate
$ python /home/uon/Documents/Detector_Demo/trained_detr/detr_server.py

5. grasp module
저쪽컴퓨터

6. main
~/ochansol/isaac_chansol/hanhwa_grasp_doosan_robotiq.py




2026 01 21 오찬솔 메모
##############################
stage 1(down),3(up) 번만 궤적제어 적용했습니다

궤적 제어 파트 수정하면서 속도 확 늦췄습니다
아래 코드부분이 로봇 제어 명령이고,
0.2 값이 속도인데 변경시 두개 모두 같이 변경해야하고, 작을수록 빠릅니다.
안전상 0.05 이상으로 사용해주세요

self.cont.send_joint_trajectory(
                                points=joint_positions,
                                times_sec=np.linspace(0.2, len(joint_positions)*0.2, len(joint_positions)),
                                include_start_from_current=True, 
                                start_hold_sec=0.1,
                                wait_result=False,
                                degrees=True,
                            )

일반 ik 제어는 
self.cont.movej(pos= joint_position,
                            vel=10.0,
                            sync_type=1)


260122 정지훈
- 통합
* detector (ROS)
input : rgb(Image), timestamp(str)
output : timestamp(empty일 경우에만 "Empty"를 뱉어 빈 박스 확인)

* grasp prediction
리스트 형식으로 여러 개 받도록 변경


* 확인
import random
depth_top_key = random.choice(list(class_dict.keys()))




==================v2
v2 장점 : pcd로 주변환경과 충돌까지 고려한 파지점 획득 가능. 모델 신뢰도가 더 높음
단점 : 객체로 인식하지 않아서 여러 개를 동시에 잡힐 수 있음. 특히 박스 뭉쳐있으면 중간 지점을 파지하는 경향

- segmentation
- graspnet
- filtering
1) segmentation 밖에 있는 점은 제외 v
2) 최우선순위 파지점 선택 : scores로 줄세웠음. v
***(((메인에서는 depth 가 가장 높은 곳(depth top key 쓸 필요?))))
3) 최우선순위 파지점이 속한 class 확인 v
***4) 각 mask 중심점이랑 파지점 거리가 가까울수록 높은 rank? -> bbox 사용? bbox의 중심점 == mask 중심점 보장되는가?
***5) 물건별로 제한된 width로 설정???
***6) 근처에 박스(또는 물체)가 있으면 근처 점 sampling해서 push

그리퍼가 들어갈 수 있는지 확인 (찬솔p가 해놓은거? 또는 {두 손가락 점의 depth 높이} - {중심점 높이}로 제한)
n번 돌아도 못정하면 empty인지 확인하고 empty가 아니면 밀기
empty 아닌데 파지점이 모두 영역 밖 : 가장자리라 파지 x

민경프로 확인 : 20260123111420, 20260123135411

그리퍼 회전 각도 맞는지 확인(-??) v -> -atan으로 해결

ISSUE
up 다음 state4에서 무한 동작 (movej안먹음?) -> wait_result가 False -> True 하니까 됨
덜내려감 z_marzin = 0 -> -0.01 바꿔놓음
내려가다가 멈춤(inverse?) 1번
    --> [INFO] [1769146968.418877307] [move_test_client]: Trajectory result status: 4
terminate called without an active exception 


TODO
intrinsic 업데이트
박스가 같이 있으면 두개 같이 잡는 경우 있음
push 개선 : 그리퍼가 들어갈 수 있는 자리 있는지 확인 등


