#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge

import numpy as np
import math
from enum import IntEnum


class FlightPhase(IntEnum):
    INIT = 0
    TAKEOFF = 1
    NAVIGATE = 2
    REACHED = 3


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Distance from point to line segment."""
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    dot = A * C + B * D
    len_sq = C * C + D * D
    if len_sq == 0:
        return math.hypot(px - x1, py - y1)
    u = max(0.0, min(1.0, dot / len_sq))
    x = x1 + u * C
    y = y1 + u * D
    return math.hypot(px - x, py - y)


class SensorFusionModule:
    def __init__(self, logger):
        self.logger = logger
        self.depth_reliability = 0.92
        self.lidar_reliability = 0.65
        self.measurement_noise_depth = 0.2
        self.measurement_noise_lidar = 0.3

    def fuse_distance(self, d_depth, d_lidar):
        d_ok = (d_depth is not None) and (0.1 < d_depth < 30.0)
        l_ok = (d_lidar is not None) and (0.1 < d_lidar < 30.0)

        if d_ok and l_ok:
            w_d = self.depth_reliability / self.measurement_noise_depth
            w_l = self.lidar_reliability / self.measurement_noise_lidar
            fused = (d_depth * w_d + d_lidar * w_l) / (w_d + w_l)
            if abs(d_depth - d_lidar) > 2.0:
                fused = min(d_depth, d_lidar)
            return fused
        if d_ok:
            return d_depth
        if l_ok:
            return d_lidar
        return 20.0

    def update(self, d_f, d_l, d_r, l_f, l_l, l_r):
        return (
            self.fuse_distance(d_f, l_f),
            self.fuse_distance(d_l, l_l),
            self.fuse_distance(d_r, l_r),
        )


class SmartObstacleNavigator(Node):
    def __init__(self):
        super().__init__("smart_obstacle_navigator")

        self.declare_parameters("", [("flight_altitude", 5.0)])
        self.flight_altitude = float(self.get_parameter("flight_altitude").value)

        self.waypoints = [

            
        ]
        self.wp_index = 0
        self.manual_goal = False

        self.phase = FlightPhase.INIT
        self.armed = False
        self.have_local_position = False

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.offboard_counter = 0

        self.raw_depth = {"f": None, "l": None, "r": None}
        self.raw_lidar = {"f": None, "l": None, "r": None}
        self.last_depth_time = 0.0
        self.last_lidar_time = 0.0

        self.fused_front = 20.0
        self.fused_left = 20.0
        self.fused_right = 20.0
        self.fusion = SensorFusionModule(self.get_logger())

        self.filtered_center_err = 0.0
        self.prev_yaw = 0.0

        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_viz = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_cmd
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_cmd
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_cmd
        )
        self.viz_path_pub = self.create_publisher(Path, "/planned_path", qos_viz)

        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.pos_cb,
            qos_cmd,
        )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status", self.status_cb, qos_cmd
        )
        self.create_subscription(Image, "/depth_camera", self.depth_cb, qos_sensor)
        self.create_subscription(LaserScan, "/scan", self.lidar_cb, qos_sensor)

        self.create_subscription(Path, "/global_path", self.path_cb, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)

        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Navigator v5.5 Ready")

    # ---------- PATH UTILS ----------

    def get_path_error(self):
        """Distance from current pose to closest segment of remaining path."""
        if len(self.waypoints) < 2 or self.wp_index >= len(self.waypoints) - 1:
            return 0.0
        px, py = self.current_x, self.current_y
        best = float("inf")
        start = max(0, self.wp_index)
        for i in range(start, len(self.waypoints) - 1):
            x1, y1 = self.waypoints[i]
            x2, y2 = self.waypoints[i + 1]
            d = point_to_segment_distance(px, py, x1, y1, x2, y2)
            if d < best:
                best = d
        return best

    def publish_visual_path(self):
        if not self.waypoints:
            return
        
        px, py = self.current_x, self.current_y
        dists = [math.hypot(wx - px, wy - py) for wx, wy in self.waypoints]
        new_index = int(np.argmin(dists))

        if new_index > 0:
            self.waypoints = self.waypoints[new_index:]
            self.wp_index = 0


        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        for x, y in self.waypoints[self.wp_index:]:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.position.z = self.flight_altitude
            msg.poses.append(p)

        self.viz_path_pub.publish(msg)

    def path_cb(self, msg: Path):
        if self.manual_goal:
            return
        if self.phase == FlightPhase.NAVIGATE and self.wp_index < len(self.waypoints) - 1:
            return
        new_wps = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if not new_wps:
            return
        self.waypoints = new_wps
        px, py = self.current_x, self.current_y
        dists = [math.hypot(wx - px, wy - py) for wx, wy in self.waypoints]
        self.wp_index = int(np.argmin(dists))

        self.publish_visual_path()
        self.get_logger().info(f"New global path accepted at index {self.wp_index}")



        

    def goal_cb(self, msg: PoseStamped):
        self.waypoints = [(msg.pose.position.x, msg.pose.position.y)]
        self.wp_index = 0
        self.manual_goal = True
        self.phase = FlightPhase.NAVIGATE
        self.publish_visual_path()
        self.get_logger().info(
            f"Manual goal: ({self.waypoints[0][0]:.2f}, {self.waypoints[0][1]:.2f})"
        )

    # ---------- SENSOR CALLBACKS ----------

    def pos_cb(self, msg: VehicleLocalPosition):
        self.current_x = msg.y
        self.current_y = msg.x
        self.current_z = -msg.z
        self.have_local_position = True

    def status_cb(self, msg: VehicleStatus):
        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def depth_cb(self, msg: Image):
        self.last_depth_time = self.get_clock().now().nanoseconds / 1e9
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            H, W = img.shape
            l, r = int(W * 0.3), int(W * 0.7)
            h1, h2 = int(H * 0.35), int(H * 0.65)

            def ext(roi):
                flat = roi.flatten()
                v = flat[(flat > 0.1) & (flat < 30.0) & np.isfinite(flat)]
                return float(np.percentile(v, 20)) if v.size > 30 else None

            self.raw_depth["l"] = ext(img[h1:h2, :l])
            self.raw_depth["f"] = ext(img[h1:h2, l:r])
            self.raw_depth["r"] = ext(img[h1:h2, r:])
            self.update_fusion()
        except Exception:
            pass

    def lidar_cb(self, msg: LaserScan):
        self.last_lidar_time = self.get_clock().now().nanoseconds / 1e9
        try:
            ranges = np.array(msg.ranges)

            def sector(angle_rad, width_deg=35.0):
                width = math.radians(width_deg)
                i_min = int((angle_rad - width - msg.angle_min) / msg.angle_increment)
                i_max = int((angle_rad + width - msg.angle_min) / msg.angle_increment)
                i_min = max(0, i_min)
                i_max = min(len(ranges), i_max)
                seg = ranges[i_min:i_max]
                v = seg[(seg > 0.1) & (seg < 30.0) & np.isfinite(seg)]
                return float(np.percentile(v, 10)) if v.size > 5 else None

            self.raw_lidar["f"] = sector(0.0)
            self.raw_lidar["l"] = sector(math.radians(90))
            self.raw_lidar["r"] = sector(math.radians(-90))
            self.update_fusion()
        except Exception:
            pass

    def update_fusion(self):
        now = self.get_clock().now().nanoseconds / 1e9
        d = self.raw_depth if now - self.last_depth_time <= 1.0 else {"f": None, "l": None, "r": None}
        l = self.raw_lidar if now - self.last_lidar_time <= 1.0 else {"f": None, "l": None, "r": None}
        self.fused_front, self.fused_left, self.fused_right = self.fusion.update(
            d["f"], d["l"], d["r"], l["f"], l["l"], l["r"]
        )

    # ---------- CONTROL LOOP ----------

    def control_loop(self):
        if not self.have_local_position:
            return
        self.pub_offboard_mode()
        # If we were flying and got disarmed (failsafe landing), restart INIT
        if not self.armed and self.phase != FlightPhase.INIT:
            self.phase = FlightPhase.INIT
            self.offboard_counter = 0
    # <- NO return here

        



        # INIT
        if self.phase == FlightPhase.INIT:
            self.pub_setpoint(self.current_x, self.current_y, self.current_z, float('nan'))
            if self.offboard_counter == 20:
                self.pub_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0
                )

            if self.offboard_counter == 25:
                self.pub_command(VehicleCommand.VEHICLE_CMD_DO_SET_HOME, p1=1.0)



            if self.offboard_counter == 50:
                self.pub_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=1.0
                )
            if self.armed and self.offboard_counter > 60:
                self.phase = FlightPhase.TAKEOFF
            self.offboard_counter += 1
            return

        # TAKEOFF
        if self.phase == FlightPhase.TAKEOFF:
            self.pub_setpoint(
                self.current_x, self.current_y, self.flight_altitude, float('nan')
            )
            if abs(self.current_z - self.flight_altitude) < 0.3:
                self.phase = FlightPhase.NAVIGATE
            return

        # NAVIGATION
        if self.wp_index >= len(self.waypoints):
            self.pub_setpoint(
                self.current_x, self.current_y, self.flight_altitude, self.prev_yaw
            )
            return

        tx, ty = self.waypoints[self.wp_index]
        dx = tx - self.current_x
        dy = ty - self.current_y
        dist = math.hypot(dx, dy)

        # Yaw toward waypoint, smoothed
        yaw_target = math.atan2(dx, dy)
        yaw_alpha = 0.2
        self.prev_yaw = (1 - yaw_alpha) * self.prev_yaw + yaw_alpha * yaw_target
        yaw_goal = self.prev_yaw

        # Base speed
        is_corridor = (self.fused_left < 4.0) and (self.fused_right < 4.0)
        base_speed = 7.5 if is_corridor else 5.0

        # Speed gating by heading alignment
        heading_error = abs(math.atan2(dy, dx))
        if heading_error < math.radians(10):
            base_speed *= 1.0
        elif heading_error < math.radians(25):
            base_speed *= 0.7
        else:
            base_speed *= 0.45

        # Obstacle slowdowns
        if self.fused_front < 4.5:
            base_speed *= 0.75
        if self.fused_front < 2.5:
            base_speed *= 0.5

        # Close to waypoint: slow but not snail
        if dist < 2.0:
            base_speed = max(base_speed * 0.4, 1.8)

        # Path error → lateral gain weight
        path_err = self.get_path_error()
        tol1, tol2 = 2.5, 4.5
        if path_err < tol1:
            path_weight = 0.0
        elif path_err > tol2:
            path_weight = 1.0
        else:
            path_weight = (path_err - tol1) / (tol2 - tol1)
        kp_side = 0.1 + 0.4 * path_weight  # 0.1–0.5

        # Direction
        if dist > 0.1:
            dir_x = dx / dist
            dir_y = dy / dist
        else:
            dir_x = dir_y = 0.0

        # Corridor centering
        center_err = self.fused_left - self.fused_right
        center_err = max(min(center_err, 2.0), -2.0)
        alpha = 0.25
        self.filtered_center_err = (
            1 - alpha
        ) * self.filtered_center_err + alpha * center_err
        side_vel = kp_side * self.filtered_center_err
        side_vel = max(min(side_vel, 2.0), -2.0)

        # Hard collision safety
        if self.fused_front < 1.2:
            base_speed = 0.0
            if self.fused_left > self.fused_right:
                side_vel = max(side_vel, 1.2)
            else:
                side_vel = min(side_vel, -1.2)
        if self.fused_front < 0.7:
            side_vel = 2.0 if self.fused_left > self.fused_right else -2.0

        tan_x = -dir_y
        tan_y = dir_x

        dt = 0.1
        nx = self.current_x + (dir_x * base_speed + tan_x * side_vel) * dt
        ny = self.current_y + (dir_y * base_speed + tan_y * side_vel) * dt
        nz = self.flight_altitude

        # Waypoint reached?
        if dist < 1.0:
            self.wp_index += 1
            self.publish_visual_path()
            if self.wp_index >= len(self.waypoints):
                self.phase = FlightPhase.REACHED

        self.pub_setpoint(nx, ny, nz, yaw_goal)
        self.offboard_counter += 1

    # ---------- PX4 HELPERS ----------

    def pub_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def pub_setpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position = [y, x, -z]  # map→PX4 NED swap
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def pub_command(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.command = cmd
        msg.param1 = p1
        msg.param2 = p2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SmartObstacleNavigator()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Crashed: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
