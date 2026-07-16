# Software

Install conda envs using the yaml config files
* `ume/conda_envs/ume_redis.yaml` (robot, teleoperation, inference)
* `ume/conda_envs/ume_torch.yaml` (training, inference)

## 1. Setup

Setup python environment variable.
```
export PYTHONPATH=.
```

### 1.1 Connect to Exoskeleton Yahboom IMU
To use Yahboom IMU, first add yourself to the dialout group to enable serial reading. (I recommend using arduino ide or vscode serial monitor to monitor serial device output)
```
bash ume/robot/ybimu/add_dialout_group.sh
```
If you still can't see serial output, then try disable brltty and reboot.

```
bash ume/robot/ybimu/disable_brltty.sh
```

Now you should be able to see serial output. And run the Yahboom official IMU reader code.
```
conda activate ume_redis
python ume/robot/ybimu/YbImuSerialLib.py
```

### 1.2 Connect to Hexfellow PCW-25 and Damiao Motors
Enable can, (make sure mobile base is on CAN0&1, arm on CAN2&3, exo on CAN4&5)
```
bash ume/robot/mobile_openarm/enable_can.sh
```
You can run this script to check which device is connected to which CAN. This script will make the damiao motors led light go green.
```
python ume/robot/ume/v6_bimanual/motor_chain_test.py
```

### 1.3 Hexfellow PCW-25 Robot Wheel Zeroing
1. connect a xbox gamepad to the computer
2. run `ume/robot/mobile_openarm/hexfellow_base_xbox_teleop.py` to start teleoperating the mobile base
3. adjust `rev_offset` in the `get_actual_position_rad` of `ume/robot/mobile_openarm/ hexfellow_base_imu_teleop_redis.py` base on your wheel mounting orientation
4. adjust `rev_offset` until the robot moves in the correct orientation (pitch corresponds to robot local frame x direction, roll is y, and yaw is theta)

## 2. Gravity compensation Demo
To help make sure UME is reproduced correctly, you can run the gravity compensation demo to check if the torque value is correct. 

**First time reproduction note**: it is safer to enable the mujoco visualization window, and multiply 0.5 to the gravity compensation torque.
```
python ume/robot/ume/v6_imu/real_teleop_sim.py
```

## 3. Teleop OpenArm

Run the arm teleoperation demo. This helps you make sure both openarm and UME are configured correctly.
* press `b` to slowly move the arm to the desired position
* press ` to engage torque-feedback teleoperation
```
process-compose -f ume/robot/ume/v6_imu/teleop_openarm.yaml
```

Run the mobile teleoperation demo. Tilt your body to control the mobile base, same keys as above to start the arm teleoperation.
```
process-compose -f ume/robot/ume/v6_imu/teleop_openarm_mobile.yaml
```

Now UME and Hexfellow OpenArm mobile manipulator are all working properly. Let's collect some data with it.

# Data Collection

Run the data collection script.
```
process-compose -f ume/robot/mobile_openarm/collect_data.yaml
```
You should see the live camera preview show up, and a window of a pure green image to indicate recording is not started (red if recording).
* press `b` to slowly move the arm to the desired position
* press ` to engage torque-feedback teleoperation
* press `u` to start recording
* press `q` to stop recording

data collected should be in the `data` folder in the format of
```
data
|--collecting
   |--YYYY_MM_DD_HH_MM_SS_TIMEZONE
      |--low_dim_npys
      |--camera_head.mp4
      |--camera_wrist_left.mp4
      |--camera_wrist_right.mp4
```

# Model Training
Train a UME model:
```
python ume/learning/mobile_openarm_ume/act_train_ume.py
```
Train No-Torque baseline model:
```
python ume/learning/mobile_openarm_no_torque/act_train_no_torque.py
```
Train UMI baseline model:
```
python ume/learning/mobile_openarm_umi/act_train_umi.py
```

# Model Evaluation on Real Robot

## 1. Setup
1. setup model ckpt path at `ume/learning/mobile_openarm_ume/model_inference_worker.py`
2. setup initial arm configuration path at `ume/learning/mobile_openarm/eval_openarm_controller.py`

## 2. Eval
Evaluate a UME model:
```
process-compose -f ume/learning/mobile_openarm_ume/eval_mobile_openarm.yaml
```
You can use similar commands to evaluate No-Torque and UMI baseline models.