# Autonomous UAV Navigation System

Autonomous UAV navigation system featuring real-time 3D occupancy-grid mapping, A* global path planning, and depth + LiDAR sensor fusion for dynamic obstacle avoidance in complex urban environments. Fully integrated with ROS 2, PX4 Offboard control, and Gazebo simulation for high-fidelity testing and deployment.

<p align="left"> <img src="https://img.shields.io/badge/ROS%202-Jazzy-blue.svg"/> <img src="https://img.shields.io/badge/PX4-Offboard-red.svg"/> <img src="https://img.shields.io/badge/Gazebo-8.10.0-orange.svg"/> <img src="https://img.shields.io/badge/Path%20Planning-A*%20(3D)-brightgreen.svg"/> <img src="https://img.shields.io/badge/Sensor%20Fusion-LiDAR+Depth-green.svg"/> <img src="https://img.shields.io/badge/Real--time%20Mapping-3D%20Voxels-purple.svg"/> </p>


## System Capabilities
| Capability                    | Technical Summary                                                                                                                   |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Mapping**                   | Height-aware voxel occupancy grid updated with Bresenham ray tracing and yaw-rate filtering to prevent distortion during rotation.  |
| **Global Planning**           | Continuous A* replanning with safety-inflated costmap and smoothed trajectory publishing.                                           |
| **Local Navigation & Safety** | Reactive velocity commands, corridor-centering and collision overrides based on depth and LiDAR fusion.                             |
| **Flight Control**            | PX4 Offboard position setpoints applied through ROS 2 middleware for stable navigation.                                             |
| **Visualization**             | 3D voxel map + global path + UAV pose in RViz for interpretability of autonomy behavior.                                            |


## Directory Structure

<img width="610" height="852" alt="Screenshot from 2025-12-10 17-17-23" src="https://github.com/user-attachments/assets/b7a45027-76ce-4bb7-832e-dc78bbceacd6" />

## Architecture
### Functional Overview

<img width="2752" height="1536" alt="3D Occupancy Grid" src="https://github.com/user-attachments/assets/0757d913-5961-4b91-aabd-1bac0986319c" />

## ROS 2 Node Overview

This project is organized into clear perception, planning and control modules. Each node runs independently and communicates over ROS topics for robust modularity.

| Node                      | Purpose                                                                    | Key Inputs                                                  | Key Outputs                                  |
| ------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------- |
| `mapping_node.py`         | Builds a live 3D voxel occupancy grid using LiDAR/Depth rays and UAV pose  | `/scan`, `/depth_camera`, `/fmu/out/vehicle_local_position` | `/map` (OccupancyGrid), 3D voxel MarkerArray |
| `path_planner.py`         | A*-based global route generation & dynamic replanning based on updated map | `/map`                                                      | `/global_path` (nav_msgs/Path)               |
| `autonomous_navigator.py` | Global path following with lateral corrections & corridor centering        | `/global_path`, fused safety distances                      | `/trajectory_setpoint` to PX4                |
| `depth_avoidance.py`      | Fuses short-range LiDAR + Depth sensing & triggers collision overrides     | `/scan`, `/depth_camera`                                    | Fused left/right/front distances for safety  |
| `offboard_control.py`     | Sends PX4 Offboard commands (arming, takeoff, set-mode, setpoints)         | Internal                                                    | Vehicle commands                             |
| `bridge_node.py`          | Coordinate frame conversion for PX4 NED ↔ ROS ENU                          | `/vehicle_local_position`                                   | TF: `map → base_link`                        |
| `sensor_test.py`          | Utility for visualizing sensor data                                        | `/scan`, `/depth_camera`                                    | Rviz debug views                             |


Nodes are launched together via:
```
ros2 launch autonomous_drone drone.launch.py
```

## ROS Data Flow
Below is the real signal wiring used in this navigation stack:
| Publisher    | Topic                             | Subscriber             | Role                           |
| ------------ | --------------------------------- | ---------------------- | ------------------------------ |
| Depth camera | `/depth_camera`                   | Mapping, Sensor Fusion | near-range depth vision        |
| LiDAR        | `/scan`                           | Mapping, Sensor Fusion | 360° obstacle geometry         |
| PX4          | `/fmu/out/vehicle_local_position` | Mapping, Navigator     | UAV pose for mapping & control |
| Mapping      | `/map`                            | A* Planner             | environmental occupancy        |
| A* Planner   | `/global_path`                    | Navigator              | waypoint tracking              |
| Navigator    | `/trajectory_setpoint`            | PX4 Flight Controller  | direct UAV command             |

### Data update rates:
/depth_camera ~30 Hz

/scan ~10 Hz

Setpoints: 10–20 Hz

Replanning whenever deviation or map changes


## Algorithms & Techniques

### Sensor Fusion (depth_avoidance.py)

Combines depth + LiDAR with reliability weighting

Median-based filtering to reject noise

Safety overrides when distance < thresholds

Generates front/left/right clearance corridors

### 3D Mapping (mapping_node.py)

Ray-tracing from sensor origin to detected hit

Voxel height stacking → vertical building shapes

Ghost clearing to erase outdated obstacles

Rotation-aware filtering reduces distortion at turns

### Global A* Planning (path_planner.py)

Real-time A* search over occupancy grid

Dynamic replanning if path blocked

Cost inflation around obstacles for clearance safety

### Local Navigation (autonomous_navigator.py)

Path tracking + lateral centering in corridors

Adaptive forward speed based on heading alignment

Collision stop-and-slide behavior when blocked

Smooth yaw blending to reduce oscillation


### PX4 Offboard Flight Control

Direct setpoints in ROS → PX4 NED

Automated arming, set-home, takeoff, landing fail-safe

## Installation & Setup

### Dependencies

ROS 2 Jazzy

PX4 Firmware (Offboard enabled)

GZ Sim 8.10.0

Python 3.10+

### Create & Prepare Workspace
```
# Create a fresh ROS 2 workspace (or use your existing one)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```
### Clone the Repository
```
git clone https://github.com/Ajinkya-001/autonomous-uav-navigation-system.git
cd ~/ros2_ws
```
### Install Dependencies
```
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -y

```


### build 
```
cd ~/px4_ros2_ws   # replace with your workspace path
colcon build --symlink-install
source install/setup.bash
```

### Launch Simulation
```
ros2 launch autonomous_drone drone.launch.py
```


## How It Works (Navigation Flow)

1. User clicks a 2D Goal Pose in RViz — sends /goal_pose.

2. UAV begins motion using local fused obstacle avoidance while the map is still unknown.

3. Once a usable map is built, the A* planner generates a global route.

4. UAV follows global path, but switches back to avoidance if blocked.

5. UAV slows down and stops once the goal is reached (<1 m proximity).

The system continuously switches between global planning and local safety behaviors depending
on environmental certainty.


## Configuration Parameters (so users can tune your system)

| Parameter              | Default         | Purpose                                  | Location                   |
| ---------------------- | --------------- | ---------------------------------------- | -------------------------- |
| `flight_altitude`      | **5.0 m**       | Cruise height for waypoint tracking      | SmartObstacleNavigator     |
| `map_resolution`       | **0.10 m**      | Grid resolution for occupancy mapping    | SimpleMapper (resolution)  |
| `safety_radius`        | **0.6 m**       | Clearance from obstacles used by A*      | DynamicPlanner params      |
| `replan_rate`          | **2.0 Hz**      | Frequency of global A* replanning        | DynamicPlanner params      |
| `base_speed`           | **5.0–7.5 m/s** | Forward velocity based on corridor width | Local navigation logic     |
| `front_stop_threshold` | **1.2 m**       | Collision emergency stop distance        | Obstacle fail-safe logic   |


## Known Limitations

Mapping may distort slightly during **rapid yaw rotation** before pose stabilization.

Requires both **LiDAR and depth camera** views overlapping for best reconstruction.
  
A* replanning frequency can increase in **dense obstacle clusters**, affecting speed.
  
System currently tested only in **PX4 SITL simulation** (no GPS fusion enabled).
 
Narrow vertical gaps may not be detected if outside depth/scan field-of-view.

## Future Work

Hardware validation on real UAV (Jetson + PX4 flight controller)
Outdoor support with **GPS + IMU fusion**
Semantic layer in mapping (roads, buildings, no-fly zones)
Moving obstacle tracking and dynamic path prediction
Multi-UAV coordination for collaborative exploration

## License

This project is licensed under the **MIT License**.

You are free to use, modify and distribute this software for research and development purposes.

## Contact

**Ajinkya Patil** 

B.Tech – Artificial Intelligence & Robotics  

Email: ajinkyapatilckl@gmail.com  

GitHub: https://github.com/Ajinkya-001


















