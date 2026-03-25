# UR5e-YOLO-Vision-Integration


# Steps to launch the system
- Launch the Camera 
  - ros2 launch realsense2_camera rs_pointcloud_launch.py
  - ALTERNATIVE: ros2 launch realsense2_camera rs_pointcloud_launch.py rgb_camera.color_profile:=1280x720x6
  - IMAGE TOPIC: /camera/camera/color/image_raw 
  - FUNCTIONS: start the RealSense camera and publish the color image stream to the topic /camera/camera/color/image_raw for the YOLO-based object detection node to subscribe.
  
- Launch UR5e
  - ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=158.125.191.88 
  - NETWORK: Lboro University IP, 158.125.191.88; robot24: 10.0.0.100
  - FUNCTIONS: establish communication between the UR5e robot and ROS2, allowing for control commands to be sent to the robot and feedback to be received.

- Launch the YOLO-based object detection node
  - ros2 run yolov5_ros yolov5_ros --ros-args --remap raw:=/camera/camera/color/image_raw 
  - ALTERNATIVE: ros2 launch yolov5_ros2 yolov5_ros2
  - FUNCTIONS: subscribe to the camera image topic, perform object detection using the YOLO algorithm, and publish the detection results (including the confidence of the target object) into a topic for the main node to subscribe.
  

- Bridge the detector and the main node
  - ros2 run confid_subpub confid_subpub
  - publish the measureemnts into: /confidence 
  - FUNCTIONS: subscribe to the detection results, extract the confidence of the target object, and publish the confidence value into the topic /confidence for the main node to subscribe.

- Launch the main node for the system
  - launch matlab: matlab -softwareopengl
  - run the main script: ur5e_ros2_lookAt_control.m
  - ros2 node publishes topic into: /urscript_interface/script_command 
  - FUNCTIONS: subscribe to the confidence topic, compute the control command based on the confidence value, and publish the control command into the topic /urscript_interface/script_command for the UR5e robot to execute.