
 # **Complete ROS 2, RViz and Gazebo roadmap: Weed Detection & Removal Robot**



Intermediate project

Simulation-first development



The goal is to develop a complete autonomous agricultural robot in simulation that can:

1. Move through a virtual agricultural field.
2. Detect weeds using a simulated camera and computer vision.
3. Estimate each weed's location.
4. Navigate towards a selected weed.
5. Align its tool with the target.
6. Activate a simulated removal mechanism.
7. Record the result and continue to the next target.

You will develop the robot's software and virtual hardware before connecting real components.

[Carbon Robotics' AI instantly IDs weeds without retraining | The Tech Buzz](https://images.openai.com/static-rsc-4/5QkYYNSy-W0LuhkDR2xgVOjaKhiXAAyzhz7F_LvqLRAjzPlyl9OeGvT-wfp56oa5cle_GtaK1sf5j7F11-_Q3kvrlz4M4j81Zeo1nGONvvX-waWmNpafQQxQD4ISO6MXvA96lznQZu7djBs3HJqQc_WojcbF2o-MSrroJ-9qCWY?purpose=inline)

[Task 2: Weed Detection - Pan African Robotics Competition](https://images.openai.com/static-rsc-4/wFS9-fXNHDzHBRqqGvOuG8e3qOCr5qoJjQDrhF0bglHI1_jlWRk6kzMmmrNbC3EhRoiH0HHNp4zsVdRNiYL3eM1XpCcg224XUHBtOZb6lKQR0pmQTb2rMcxUTNzt9WvhueCcLF-l8zuUKwc1SQLCZ6YmWHepw0xjRFdobsmdL4w?purpose=inline)

[JetAuto: Build Robotic SLAM Function with Lidar and Depth Camera  – Hiwonder](https://images.openai.com/static-rsc-4/tLsgJ6uA_3KsW6uJHjbFEblHmUQSowWn66-UMDPjqNHPLPGkLv9tCB-6EfUdouzEE2D6Zo2OZ3X3tP_bVlANS0BBGCzHP6fSzWtgvHDN9J9sIcNqbbABos2QuDj9srQ9mwfij_ZnzpLJGjGmsiifn8LhTFKyEy4hyoUH_aSO7hE?purpose=inline)

10

## **1. Final project architecture**

I recommend starting with a four-wheel skid-steer ground robot. It is a practical architecture for a small agricultural rover.

## **WeedBot — system architecture**

**Perception subsystem**

RGB camera · Depth camera · OpenCV / YOLO

Weed detection and target coordinates

**ROS 2 decision-making**

Detection processing · Target selection · Task manager

**Navigation**

Nav2 · SLAM · Path planning

**Removal tool**

Target alignment · Simulated actuator

**Gazebo Harmonic simulation**

Robot physics · Field · Weeds · Obstacles · Sensors

RViz 2 visualises robot state, maps, detections, sensor data and navigation paths.

### **Main hardware subsystems to represent**

| Subsystem        | Simulated component              |
| ---------------- | -------------------------------- |
| Chassis          | Four-wheel mobile platform       |
| Drive            | Four wheel joints, skid-steer    |
| Vision           | Front-facing RGB-D camera        |
| Obstacle sensing | 2D LiDAR                         |
| Orientation      | IMU                              |
| Position         | Wheel odometry                   |
| Computing        | Virtual ROS 2 computer           |
| Removal tool     | Non-emitting simulated actuator  |
| Power            | Virtual battery state (optional) |

**Important design boundary:** In the simulation, represent the removal tool as a virtual actuator or marker, not an operating high-power laser. Do not use the simulation as proof of real-world laser safety. For a physical agricultural machine, prioritise a guarded mechanical removal tool or a professionally engineered, certified and interlocked system.

## **2. Software and installation**

Recommended software combination

Use this consistent software stack rather than mixing unrelated ROS and Gazebo versions.

| Software          | Version               |
| ----------------- | --------------------- |
| Operating system  | Ubuntu 24.04 LTS      |
| ROS 2             | Jazzy Jalisco         |
| Simulator         | Gazebo Harmonic       |
| Visualisation     | RViz 2                |
| Programming       | Python 3, C++         |
| Computer vision   | OpenCV, optional YOLO |
| Navigation        | Nav2                  |
| Robot description | URDF / Xacro          |
| Build system      | colcon                |

Gazebo's installation guidance recommends Ubuntu 24.04, ROS 2 Jazzy and Gazebo Harmonic as a compatible combination.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+2



### **Step 1 — Install Ubuntu**

If your computer currently runs Windows, you can use Ubuntu 24.04 in a separate installation or a suitable virtual machine. For Gazebo's 3D simulation, a native Ubuntu installation generally offers a better experience than a resource-constrained virtual machine.

Check your system:

```
lsb_release -a
uname -m
```

### **Step 2 — Install ROS 2 Jazzy**

Follow the official installation instructions:

[ROS 2 Jazzy — Ubuntu installation](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html?utm_source=chatgpt.com)

After setting up the ROS 2 apt repository, install the desktop version:

```
sudo apt update
sudo apt install ros-jazzy-desktop
```

Source ROS 2:

```
source /opt/ros/jazzy/setup.bash
```

Check:

```
ros2 --version
ros2 topic list
```

Add automatic sourcing to your terminal:

```
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### **Step 3 — Install Gazebo Harmonic and ROS integration**

```
sudo apt install ros-jazzy-ros-gz
```

This installs the Gazebo integration packages appropriate for your ROS distribution. The `ros_gz_bridge` package connects ROS 2 messages with Gazebo Transport messages.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://gazebosim.org\&sz=32)

Gazebo harmonic documentation+1



Test Gazebo:

```
gz sim
```

### **Step 4 — Install navigation, perception and development packages**

```
sudo apt install \
ros-jazzy-navigation2 \
ros-jazzy-nav2-bringup \
ros-jazzy-slam-toolbox \
ros-jazzy-xacro \
ros-jazzy-robot-state-publisher \
ros-jazzy-joint-state-publisher-gui \
ros-jazzy-cv-bridge \
ros-jazzy-image-transport \
ros-jazzy-vision-msgs \
ros-jazzy-tf-transformations \
python3-opencv \
python3-colcon-common-extensions \
python3-rosdep \
python3-vcstool \
git
```

Install optional control packages as well:

```
sudo apt install \
ros-jazzy-ros2-control \
ros-jazzy-ros2-controllers
```

You may use Gazebo's own drive plugin initially and add ROS 2 Control when you want more realistic controller integration.

### **Step 5 — Test RViz and ROS 2**

```
rviz2
```

You should see the RViz window.

Check the ROS environment:

```
printenv | grep ROS
```

You should see `ROS_DISTRO=jazzy`.

## **3. Create your ROS 2 project workspace**

We will keep each part of the robot in a separate ROS 2 package. This makes the project easier to debug, extend and publish on GitHub.

### **Step 6 — Create the workspace**

```
mkdir -p ~/weedbot_ws/src
cd ~/weedbot_ws/src
```

Create the packages:

```
ros2 pkg create weedbot_description \
  --build-type ament_cmake

ros2 pkg create weedbot_sim \
  --build-type ament_python \
  --dependencies rclpy geometry_msgs sensor_msgs nav_msgs

ros2 pkg create weedbot_perception \
  --build-type ament_python \
  --dependencies rclpy sensor_msgs cv_bridge vision_msgs geometry_msgs

ros2 pkg create weedbot_navigation \
  --build-type ament_python \
  --dependencies rclpy geometry_msgs nav2_msgs

ros2 pkg create weedbot_tasks \
  --build-type ament_python \
  --dependencies rclpy geometry_msgs tf2_ros

ros2 pkg create weedbot_bringup \
  --build-type ament_python \
  --dependencies rclpy
```

Build:

```
cd ~/weedbot_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Your initial package architecture will look like this:

**weedbot_ws/**

**weedbot_description/**

URDF, Xacro, meshes and robot dimensions

**weedbot_sim/**

Gazebo world, models and simulation configuration

**weedbot_perception/**

Camera, weed detection and target coordinates

**weedbot_navigation/**

Nav2, SLAM, maps and robot footprint

**weedbot_tasks/**

Target selection and removal task manager

**weedbot_bringup/**

Launch files and complete system startup

## **4. Robot modelling: URDF and Xacro**

This is where your Solid Edge knowledge becomes useful.

### **Step 7 — Design the robot in Solid Edge**

Start with a basic mechanical assembly.

[Free CAD Designs, Files & 3D Models | The GrabCAD Community Library](https://images.openai.com/static-rsc-4/oK5Vzlz_a7kbntkRxHC1fOXs8looWYICDSUZ8EbJleKp5LoVQ4xTFQ1gojjFIC3SFWeQffWqqqZZqouvyiLTGpm0oulEZfGRoGPS2ZAyWUEEpUEGjJSEuGbDeKimSH_LhZfqM2mAyaUJHdJ75eZelmVGNh-lYxDRHEzr7Xl8bvM?purpose=inline)

[Comparative Study on Simulated Outdoor Navigation for Agricultural Robots](https://images.openai.com/static-rsc-4/aa1OfuAAZl_0VWsF9RrkWhDkcVUTjL_a_C3RUUIB9cZ-OvRuiUbG0dj0RreWPsOS15ezzcyM1gxpnG0SqDgW_JD10yEQTKJzkNwMyXw47wBa4gXiChje4aBUT3212nrc-yNFCTBojiLjkW6pKcjJgmHdxbv8Wu7yRO5KbMC1uJ0?purpose=inline)

[Free CAD Designs, Files & 3D Models | The GrabCAD Community Library](https://images.openai.com/static-rsc-4/u4S6zLvDIPxolto_olMtB0J-_IYrYVijJuSQWRBjZqXHEVaDspJnxNrw5uux_CTDtDcNdCBOIVj7f-exMdusqF9ly0LFL1U5gxOPaKtaSM_Lce0hKjHVUFxAGgX3Nx7CJnxLNi5WWz06zL4OoPCYQw2Py4pXNZs9ntCqFwFmkg8?purpose=inline)

10

Recommended starting dimensions (design assumptions, not measured specifications):

| Component              | Initial dimension |
| ---------------------- | ----------------- |
| Chassis length         | 500 mm            |
| Chassis width          | 400 mm            |
| Chassis height         | 120 mm            |
| Wheel diameter         | 150 mm            |
| Wheel width            | 50 mm             |
| Ground clearance       | 80 mm             |
| Camera mounting height | 450 mm            |

Create these CAD parts:

1. Main chassis.
2. Four wheels.
3. Four motor mounting brackets.
4. Camera mounting bracket.
5. LiDAR mounting bracket.
6. Electronics enclosure.
7. Battery enclosure.
8. Removal-tool mounting assembly.

Export individual components as STL or suitable mesh files for simulation. Keep the editable Solid Edge parts and assemblies as your engineering source files.

### **Step 8 — Understand the URDF robot structure**

URDF stands for Unified Robot Description Format. It describes the robot's links, joints, coordinate frames, visual geometry and collision geometry.

Your robot tree should look like this:

**Robot TF / link structure**

`base_link`

`base_footprint`

`imu_link`

`camera_link`

`lidar_link`

front_left_wheel

front_right_wheel

rear_left_wheel

rear_right_wheel

### **Step 9 — Create the initial Xacro model**

Create:

`weedbot_description/urdf/weedbot.urdf.xacro`

Here is a simplified starter model:

```

<?xml version="1.0"?>
<robot name="weedbot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <xacro:property name="chassis_length" value="0.50"/>
  <xacro:property name="chassis_width" value="0.40"/>
  <xacro:property name="chassis_height" value="0.12"/>
  <xacro:property name="wheel_radius" value="0.075"/>

  <link name="base_footprint"/>

  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.12"/>
      <geometry>
        <box size="${chassis_length} ${chassis_width} ${chassis_height}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.12"/>
      <geometry>
        <box size="${chassis_length} ${chassis_width} ${chassis_height}"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.12"/>
      <mass value="12.0"/>
      <inertia ixx="0.20" ixy="0" ixz="0"
               iyy="0.25" iyz="0" izz="0.30"/>
    </inertial>
  </link>

  <joint name="base_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
    <origin xyz="0 0 0"/>
  </joint>

  <link name="camera_link"/>
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.22 0 0.40" rpy="0 0 0"/>
  </joint>

  <link name="lidar_link"/>
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 0.25"/>
  </joint>

  <!-- Add four wheel links and continuous wheel joints here. -->
  <!-- Add the IMU frame and camera optical frame. -->
  <!-- Add Gazebo sensor and drive configuration separately. -->

</robot>

```

This is a starting fragment, not a complete simulation-ready robot. You must add the four wheel links and joints, valid inertia values, mesh geometry, camera optical frame, and Gazebo sensor and drive configuration.

Test the URDF:

```
ros2 run xacro xacro \
  src/weedbot_description/urdf/weedbot.urdf.xacro \
  > /tmp/weedbot.urdf

check_urdf /tmp/weedbot.urdf
```

Visualise it:

```
ros2 launch urdf_tutorial display.launch.py \
  model:=/tmp/weedbot.urdf
```

Install the tutorial package if needed:

```
sudo apt install ros-jazzy-urdf-tutorial
```

## **5. Build the agricultural environment in Gazebo**

### **Step 10 — Create a virtual farm**

[Task 2: Weed Detection - Pan African Robotics Competition](https://images.openai.com/static-rsc-4/wFS9-fXNHDzHBRqqGvOuG8e3qOCr5qoJjQDrhF0bglHI1_jlWRk6kzMmmrNbC3EhRoiH0HHNp4zsVdRNiYL3eM1XpCcg224XUHBtOZb6lKQR0pmQTb2rMcxUTNzt9WvhueCcLF-l8zuUKwc1SQLCZ6YmWHepw0xjRFdobsmdL4w?purpose=inline)

[Sowbot / Feldfreund DevKit ROS — WIP open agricultural robot stack - Projects - Open Robotics Discourse](https://images.openai.com/static-rsc-4/7DxgYRbEaC8KIwg8YbO437Sqck4jBxNwdWYBXNS5eoZZhnecCaDWx2RcLSxUfB6wsnEYi1a5I0-JU1voIhTiMGkT3pFp6oXOcfCxqT1mHd5a-MhBpxVB1SkpTF3HjaX90EY_X7U1S-SjulQzvjrvG9t1SVy6edmKSKAjZgZwOKw?purpose=inline)

[Setting up your workspace - Pan African Robotics Competition 2024](https://images.openai.com/static-rsc-4/S6Atb_MtlpV_ITw8SA7HVNSBkStRa2WWtXrDHkc7_huA98MFFDVil72aAvVH2xMPXYMvP4NnKraG4-xnNYtFNFLU3lc2Nx26zTBdFw8ZI7t9usGGFIb1y7V49u6xT6NBL3LbwF8FbNOE7P088qJsW7ip2sPdNx7Ihk8siXVZgv0?purpose=inline)

9

Create a world containing:

- Flat agricultural ground.
- Crop rows.
- Weed models in different locations.
- Rocks and other obstacles.
- Robot starting location.
- Lighting and shadows.
- Defined boundaries.

Suggested world:

`weedbot_sim/worlds/farm_field.sdf`

Start with a simple flat world. Add terrain variations and crop obstacles only after the robot can navigate reliably.

### **Step 11 — Add virtual sensors**

| Sensor         | ROS 2 topic               | Purpose                   |
| -------------- | ------------------------- | ------------------------- |
| RGB camera     | `/camera/image_raw`       | Weed recognition          |
| Depth camera   | `/camera/depth/image_raw` | Distance estimation       |
| Camera info    | `/camera/camera_info`     | Camera calibration        |
| LiDAR          | `/scan`                   | Obstacle detection        |
| IMU            | `/imu/data`               | Orientation               |
| Wheel odometry | `/odom`                   | Robot position estimation |

Gazebo sensors publish simulation data. Use `ros_gz_bridge` to make the appropriate sensor and control topics available in ROS 2. Check the actual topic names generated by your Gazebo model rather than assuming they match the examples.

Useful commands:

```
gz topic -l
```

```
ros2 topic list
```

```
ros2 topic echo /scan
```

### **Step 12 — Connect Gazebo and ROS 2**

The bridge is the communication layer.

Example bridge command for the simulation clock:

```
ros2 run ros_gz_bridge parameter_bridge \
  /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock
```

You will configure additional bridge entries for camera images, LiDAR, IMU, odometry and velocity commands. Their exact message types and directions depend on the topics and plugins in your Gazebo model.

Official reference: [Gazebo and ROS 2 integration](https://gazebosim.org/docs/harmonic/ros2_integration/?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://gazebosim.org\&sz=32)

Gazebo harmonic documentation



## **6. Weed detection using computer vision**

This is one of the most important components of your project.

Since your existing weed-recognition work involved camera input and OpenCV, I recommend keeping your recognition algorithm separate from the robot navigation software. That will make it easier to reuse and improve.&#x20;

### **Step 13 — Create the perception pipeline**

[Frontiers | YOLOX-based blue laser weeding robot in corn field](https://images.openai.com/static-rsc-4/V3fSTCqBpsJia4lnWrG4MgKbpo4FQ9gUQFK-BPtc_DGXhy0cdRZyP3eFdhyfGV4soNOS_rmWqcxqsRq0rPGf4dM2_xDqJAXY_qwG1iUuA2PCDWvvk_9HUz4suKgZb3rwb6FExHVChqz84prAlsVxAiGnYbHJoD82BEJQrjrZ_ko?purpose=inline)

[frontiersin.org](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.1017803/full?utm_source=chatgpt.com)

The pipeline:

1. Receive RGB camera frames.
2. Preprocess the image.
3. Detect weeds.
4. Generate bounding boxes.
5. Estimate depth at each detection.
6. Calculate the 3D target location.
7. Publish the target location to ROS 2.

For initial testing, use OpenCV colour segmentation or a simple trained model. Later, integrate YOLO for more reliable classification.

### **Step 14 — Camera detection node**

Create:

`weedbot_perception/weedbot_perception/weed_detector.py`

Basic image-processing example:

```
            hsv,            lower_green,            upper_green        )        contours, _ = cv2.findContours(            mask,            cv2.RETR_EXTERNAL,            cv2.CHAIN_APPROX_SIMPLE        )        for contour in contours:            area = cv2.contourArea(contour)            if area > 150:                x, y, w, h = cv2.boundingRect(contour)                cv2.rectangle(                    frame,                    (x, y),                    (x + w, y + h),                    (0, 255, 0),                    2                )        cv2.imshow('Weed Detection', frame)        cv2.waitKey(1)def main(args=None):    rclpy.init(args=args)    node = WeedDetector()    try:        rclpy.spin(node)    finally:        node.destroy_node()        cv2.destroyAllWindows()        rclpy.shutdown()if __name__ == '__main__':    main()
```

This is only a visual prototype. Green colour segmentation cannot reliably distinguish weeds from crops, so replace it with a trained classifier or object-detection model for meaningful performance testing.

### **Step 15 — Convert a detected weed into a 3D position**

A bounding box alone gives you a pixel location, not a physical target position.

For a depth camera, let:

- \\(u,v\\): centre pixel of the detection.
- \\(Z\\): depth measured at that pixel.
- \\(f_x,f_y\\): camera focal lengths in pixels.
- \\(c_x,c_y\\): camera principal point.

Calculate the camera-frame coordinates:

\\[ X = \frac{(u-c_x)Z}{f_x} \\]

\\[ Y = \frac{(v-c_y)Z}{f_y} \\]

\\[ Z = Z \\]

Then use TF2 to transform the point from `camera_optical_frame` into `map` or `base_link`.

Your target node should publish a `geometry_msgs/PoseStamped` or a custom weed-target message.

**Important:** Use a median depth from a suitable region inside the detection, not just one potentially noisy pixel. Confirm that the target corresponds to the ground location of the weed, rather than the top of its leaves.

## **7. Autonomous navigation with Nav2**

### **Step 16 — Learn SLAM and mapping**

The robot must understand its environment before navigating autonomously.

Use:

- SLAM Toolbox for mapping.
- AMCL for localisation on a saved map.
- Nav2 for planning and navigation.
- LiDAR for obstacle detection.
- Wheel odometry and IMU for motion estimation.

[ROS2로 turtlebot3 제어하기 3장. Nav2 이해하기와 파이썬 환경설정](https://images.openai.com/static-rsc-4/FO0nbCGXsJ5w08lccnCwpIQl3Qzld6z9wYSN5wKIeoQOg3AmfyrsGF6Ni7U4WyDbTeAhlpLlyBYnLDbwRMXmrvPxF5BRFygebn0MK5z7MWHnfRWm3eY0mFblANri-geMd1ST7ac7rkGn7qp52sMY_RD6GtalVpJy78d_TPNKaV0?purpose=inline)

[SLAM | Husarion](https://images.openai.com/static-rsc-4/pfhOGlz-Pax8ciCqMfczC5Of9LMtkepcWwJDDi5TYHe0nWCfNqTPXJDEYzVLYJBy9a0SZbwg8SffuFdbACD2f13XaD9lAopwHrhK_q8GIxsnCjqdtMj7mTSc18cyLCxgnPHZIR5xapNV492YMFYDbYg-CGh1GsPuDAqk-f-QtU4?purpose=inline)

[Cannot properly visualize local costmap in rviz2 Exercise 5.3 Unit 5 - ROS2 Navigation - The Construct ROS Community](https://images.openai.com/static-rsc-4/1tr8ExmxYPZ0ldbPAtn_QFipqC0lz2qhIekyqkTASzvrC7dVY2_3bhF9pa-paeea-f7hcFqD216aDv7lXga6V8aGxlHJOHbhmi8nIJZrBwsb7hAvE9XtTHC9C-Viq_1oQDfBfhvPeJBuAbk-kbpLJ1fe6PtH-5_eU3jGEBX7EmE?purpose=inline)

7

Start with the official Nav2 first-time robot setup. It covers transforms, robot description, odometry, sensors, mapping, footprint and navigation plugins.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.nav2.org\&sz=32)

Nav2 Docs+1



[Nav2 robot setup guide](https://docs.nav2.org/jazzy/configuration_and_development/first_time_robot_setup_guide/?utm_source=chatgpt.com)

Before integrating weed detection, make sure the robot can:

- Move forward and backward.
- Rotate.
- Avoid obstacles.
- Build a map.
- Navigate to a goal position.
- Stop safely.

### **Step 17 — Test Nav2 before using your custom robot**

You can use the official TurtleBot 3 simulation as a temporary learning exercise:

```
ros2 launch nav2_bringup tb3_simulation_launch.py \
  headless:=False
```

This is a separate reference robot, not your WeedBot. It helps verify that the ROS 2, Gazebo, Nav2 and RViz environment works before debugging your own model.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://ros-navigation.github.io\&sz=32)

Nav2 Docs



Then configure Nav2 for your own robot:

- Robot footprint.
- Maximum linear and angular velocity.
- Wheel odometry.
- LiDAR input.
- Costmaps.
- Planner and controller.
- Recovery behaviours.

## **8. Target selection and simulated removal**

This is the part that makes the project more than a standard autonomous navigation robot.

### **Step 18 — Create a weed task manager**

Create a state machine:

**1. SEARCH**

Scan the field using the camera

**2. DETECT**

Identify weeds and estimate locations

**3. SELECT TARGET**

Choose a reachable weed

**4. NAVIGATE AND ALIGN**

Move to the target and check alignment

**5. SIMULATED TOOL ACTION**

Activate a virtual, non-emitting removal tool

**6. VERIFY**

Mark the target as processed in simulation

Return to SEARCH for the next target

### **Step 19 — Define the task manager logic**

The following is a conceptual implementation of the task flow, not a complete executable ROS node:

```
class WeedTaskManager:    def __init__(self):        self.state = "SEARCH"        self.targets = []        self.current_target = None    def process_detections(self, detections):        if self.state == "SEARCH":            self.targets = detections            if self.targets:                self.state = "SELECT"    def select_target(self):        if self.state == "SELECT":            self.current_target = self.targets[0]            self.state = "NAVIGATE"    def navigation_completed(self):        if self.state == "NAVIGATE":            self.state = "ALIGN"    def alignment_completed(self):        if self.state == "ALIGN":            self.state = "SIMULATED_TOOL_ACTION"    def tool_action_completed(self):        if self.state == "SIMULATED_TOOL_ACTION":            self.state = "VERIFY"
```

In the real ROS 2 implementation, use Nav2 action clients for navigation, feedback from the robot's sensors for alignment, and a simulated actuator action with explicit success/failure results.

Do not simply use a fixed timer to assume the weed has been removed. The simulation should verify a tool-action result and update the virtual target state.

## **9. Complete simulation launch**

Once the individual packages work, combine them using one launch file.

The overall launch should start:

1. Gazebo farm environment.
2. Robot model spawning.
3. `robot_state_publisher`.
4. ROS-Gazebo bridge.
5. Camera and LiDAR data.
6. Weed detection node.
7. Target coordinate node.
8. Nav2.
9. Task manager.
10. RViz 2.

Example conceptual startup:

```
cd ~/weedbot_ws

source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch weedbot_bringup weedbot_sim.launch.py
```

The launch file will be implemented in your `weedbot_bringup` package. You should only use this command once that file and all its dependencies exist.

## **10. Testing and performance validation**

Do not consider the project complete just because the robot moves in Gazebo.

Test each subsystem independently and then test the entire autonomous workflow.

| Test               | What to measure             | Success condition              |
| ------------------ | --------------------------- | ------------------------------ |
| Robot model        | URDF and collision geometry | Loads without errors           |
| Drive              | Linear and angular motion   | Responds correctly             |
| Camera             | Image stream                | Stable image data              |
| LiDAR              | Scan stream                 | Correct obstacle measurements  |
| Detection          | Precision, recall and F1    | Measured on a labelled dataset |
| Target position    | 3D position error           | Within your defined tolerance  |
| Navigation         | Goal success rate           | Reaches valid goals            |
| Obstacle avoidance | Collision events            | No collisions in test cases    |
| Tool simulation    | Correct target selection    | No false activations           |
| Full workflow      | Completed weed targets      | Consistent repeatable results  |

### **Test scenarios**

- One weed in an open area.
- Multiple weeds in different rows.
- Weeds partly hidden by crop leaves.
- Obstacles near a weed.
- Two weeds close together.
- Poor lighting.
- Camera depth noise.
- Robot unable to reach a target.
- Emergency stop during a simulated tool action.

Also save ROS 2 bag recordings for repeatable testing:

```
ros2 bag record \
  /camera/image_raw \
  /scan \
  /odom \
  /imu/data \
  /tf \
  /tf_static
```

## **11. Eight-week implementation plan**

Suggested schedule, assuming consistent practical work alongside learning.

| Week | Work                                    | Deliverable                   |
| ---- | --------------------------------------- | ----------------------------- |
| 1    | Ubuntu, ROS 2 and Gazebo installation   | Working software environment  |
| 2    | URDF, Xacro and Solid Edge robot design | Initial robot model           |
| 3    | Gazebo farm and sensor simulation       | Robot in virtual farm         |
| 4    | OpenCV / YOLO weed detection            | Weed detection node           |
| 5    | Depth and TF2 target coordinates        | 3D target positions           |
| 6    | Nav2 and autonomous navigation          | Autonomous field movement     |
| 7    | Task manager and virtual tool action    | Weed-processing workflow      |
| 8    | Testing, documentation and GitHub       | Complete simulation portfolio |

## **12. GitHub portfolio structure**

Suggested repository name:

`ros2-weed-detection-removal-robot`

```
ros2-weed-detection-removal-robot/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── docs/
│   ├── architecture.md
│   ├── installation.md
│   ├── simulation_workflow.md
│   └── testing_results.md
│
├── cad/
│   ├── solid_edge_parts/
│   ├── assemblies/
│   ├── drawings/
│   └── exports/
│
├── src/
│   ├── weedbot_description/
│   │   ├── urdf/
│   │   ├── meshes/
│   │   └── rviz/
│   │
│   ├── weedbot_sim/
│   │   ├── worlds/
│   │   ├── models/
│   │   └── config/
│   │
│   ├── weedbot_perception/
│   │   ├── weed_detector.py
│   │   ├── target_localizer.py
│   │   └── models/
│   │
│   ├── weedbot_navigation/
│   │   ├── config/
│   │   └── maps/
│   │
│   ├── weedbot_tasks/
│   │   └── task_manager.py
│   │
│   └── weedbot_bringup/
│       └── launch/
│
├── tests/
│   ├── test_detection.py
│   ├── test_target_localization.py
│   └── test_navigation.py
│
└── media/
    ├── robot_cad.png
    ├── gazebo_simulation.mp4
    ├── rviz_navigation.png
    └── weed_detection_demo.mp4
```

### **What your README should demonstrate**

- Problem statement and project objectives.
- Mechanical design and robot architecture.
- ROS 2 package architecture.
- Gazebo simulation environment.
- Weed detection and localisation method.
- Navigation and simulated removal workflow.
- Test results with actual measured metrics.
- Video demonstration.
- Installation and execution instructions.
- Limitations and future hardware integration.

A useful project title for your portfolio:

**Design and Simulation of an Autonomous Weed Detection and Precision Removal Robot Using ROS 2, Gazebo and Computer Vision.**

Do not describe the system as a physically completed autonomous robot until the hardware has actually been built and tested. Clearly label the CAD, simulation and physical implementation stages.

## **13. Your final software and hardware integration path**

Once the simulation is stable, move towards real hardware gradually.

1. **Stage 1 — Simulation**

   Ubuntu + ROS 2 + Gazebo + RViz + OpenCV
2. **Stage 2 — Embedded computer**

   Raspberry Pi 5 for camera processing and ROS 2 nodes.
3. **Stage 3 — Motor control**

   ESP32 or a suitable motor controller for wheel actuation, with a defined ROS 2 communication interface.
4. **Stage 4 — Sensor integration**

   Physical camera, IMU, LiDAR and wheel encoders.
5. **Stage 5 — Field validation**

   Controlled low-speed testing, obstacle detection, emergency stops and a safe removal-tool prototype.

## **14. Your essential reference library**

| Resource                                                                                                          | What to study         |
| ----------------------------------------------------------------------------------------------------------------- | --------------------- |
| [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/?utm_source=chatgpt.com)                                              | ROS fundamentals      |
| [Gazebo Harmonic](https://gazebosim.org/docs/harmonic/?utm_source=chatgpt.com)                                    | Physics simulation    |
| [Nav2](https://docs.nav2.org/?utm_source=chatgpt.com)                                                             | Autonomous navigation |
| [URDF tutorials](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/URDF/URDF-Main.html?utm_source=chatgpt.com) | Robot modelling       |
| [OpenCV](https://docs.opencv.org/4.x/?utm_source=chatgpt.com)                                                     | Image processing      |
| [Ultralytics YOLO](https://docs.ultralytics.com/?utm_source=chatgpt.com)                                          | Object detection      |
| [Nav2 GitHub](https://github.com/ros-navigation/navigation2?utm_source=chatgpt.com)                               | Practical ROS 2 code  |

## **15. Final project completion checklist**

**Project progress**

# **0%**

0 of 12 milestones completed

Ubuntu 24.04 and ROS 2 Jazzy installed

Gazebo Harmonic and RViz working

Four-wheel robot designed in Solid Edge

URDF / Xacro model validated

Virtual farm world created

Camera, depth, LiDAR and IMU connected

Weed detection node working

3D target coordinates calculated

Nav2 autonomous navigation working

Task manager implemented

Simulated tool action and verification working

Testing results and GitHub documentation published

Reset checklist

**My recommendation:** Start with the complete four-wheel robot model and a simple Gazebo farm. Do not begin with YOLO, Nav2 and tool automation simultaneously. First prove that your robot can spawn, move, publish odometry and display its sensors in RViz. Then add each autonomous feature one at a time.

Would you like me to check your progress every week and help you complete the next ROS 2 project milestone?

🤖Weekly ROS 2 project progress reviewTap to set up# -Real-time--Object--Detection-
Real-time object detection using Python, OpenCV and YOLO. This project detects objects from webcam/live video.
# Real-time Object Detection

This project performs real-time object detection using YOLO and OpenCV. 

## 🚀 Features
- Detect objects from webcam
- Fast and accurate detection
- Works in real-time

## 🛠️ Tech Used
- Python
- OpenCV
- YOLO

## ▶️ How to Run
1. Install dependencies:
pip install -r requirements.txt

2. Run the project:
python main.py

## 📸 Output
(Add screenshots here)

## 📌 Author
Shrinivas Masalavada
