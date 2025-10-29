# 水下 ROS 仿真平台

## 安装
1. 安装 ROS Noetic: sudo apt install ros-noetic-desktop-full
2. 克隆仓库: git clone https://github.com/uuvsimulator/uuv_simulator.git
3. cd underwater-ros-sim
4. rosdep install --from-paths src --ignore-src -y
5. catkin build
6. source devel/setup.bash

## 运行
roslaunch project_launch auv_full.launch

## 任务测试
rostopic pub /nl_command std_msgs/String "巡检 A 区"

## 迁移真实机器人
替换 Gazebo 插件为硬件驱动。

License: MIT
