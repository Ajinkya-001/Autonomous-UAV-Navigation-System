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
from collections import deque


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

        self.waypoints = []
        self.wp_index = 0
        self.manual_goal = False
        self.last_path_hash = None

        self.phase = FlightPhase.INIT
        self.armed = False
        self.have_local_position = False

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0
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
        
        # Smooth velocity control
        self.current_forward_vel = 0.0
        self.current_side_vel = 0.0
        
        # Stuck detection
        self.stuck_counter = 0
        self.position_history = deque(maxlen=50)
        
        # Performance tracking
        self.last_waypoint_time = 0.0

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

        self.get_logger().info("Navigator v7.1 Ready - Balanced & Fluent")

    # ---------- PATH UTILS ----------

    def get_path_error(self):
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

    def get_lookahead_target(self, lookahead_dist=2.5):
        if self.wp_index >= len(self.waypoints):
            return None
        
        accumulated_dist = 0.0
        px, py = self.current_x, self.current_y
        
        for i in range(self.wp_index, len(self.waypoints)):
            wx, wy = self.waypoints[i]
            seg_dist = math.hypot(wx - px, wy - py)
            
            if accumulated_dist + seg_dist >= lookahead_dist:
                remaining = lookahead_dist - accumulated_dist
                if seg_dist > 0:
                    t = remaining / seg_dist
                    return (px + t * (wx - px), py + t * (wy - py))
                return (wx, wy)
            
            accumulated_dist += seg_dist
            px, py = wx, wy
        
        return self.waypoints[-1]

    def is_stuck(self):
        if len(self.position_history) < 30:
            return False
        
        positions = list(self.position_history)
        recent = positions[-10:]
        older = positions[-30:-20]
        
        recent_center = (sum(p[0] for p in recent) / len(recent),
                        sum(p[1] for p in recent) / len(recent))
        older_center = (sum(p[0] for p in older) / len(older),
                       sum(p[1] for p in older) / len(older))
        
        movement = math.hypot(recent_center[0] - older_center[0],
                             recent_center[1] - older_center[1])
        return movement < 0.5

    def publish_visual_path(self):
        if not self.waypoints:
            return
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
        new_wps = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if not new_wps:
            return
        
        path_hash = hash(tuple(new_wps))
        if path_hash == self.last_path_hash:
            return
        self.last_path_hash = path_hash
        
        if self.manual_goal:
            self.manual_goal = False
        
        if self.waypoints and self.phase == FlightPhase.NAVIGATE:
            old_target = self.waypoints[min(self.wp_index, len(self.waypoints)-1)]
            dists_to_old_target = [math.hypot(wx - old_target[0], wy - old_target[1]) for wx, wy in new_wps]
            new_target_idx = int(np.argmin(dists_to_old_target))
            
            if dists_to_old_target[new_target_idx] < 5.0:
                self.waypoints = new_wps
                self.wp_index = new_target_idx
            else:
                self.waypoints = new_wps
                px, py = self.current_x, self.current_y
                dists = [math.hypot(wx - px, wy - py) for wx, wy in self.waypoints]
                self.wp_index = int(np.argmin(dists))
        else:
            self.waypoints = new_wps
            px, py = self.current_x, self.current_y
            dists = [math.hypot(wx - px, wy - py) for wx, wy in self.waypoints]
            self.wp_index = int(np.argmin(dists))
        
        self.publish_visual_path()
        if self.phase == FlightPhase.REACHED and abs(self.current_z - self.flight_altitude) < 0.5:
            self.phase = FlightPhase.NAVIGATE

    def goal_cb(self, msg: PoseStamped):
        self.waypoints = [(msg.pose.position.x, msg.pose.position.y)]
        self.wp_index = 0
        self.manual_goal = True
        self.phase = FlightPhase.NAVIGATE
        self.publish_visual_path()

    # ---------- SENSOR CALLBACKS ----------

    def pos_cb(self, msg: VehicleLocalPosition):
        self.current_x = msg.y
        self.current_y = msg.x
        self.current_z = -msg.z
        self.current_yaw = msg.heading
        self.have_local_position = True
        self.position_history.append((self.current_x, self.current_y))

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
        
        dt = 0.1
        self.pub_offboard_mode()
        
        if not self.armed and self.phase != FlightPhase.INIT:
            self.phase = FlightPhase.INIT
            self.offboard_counter = 0
            self.current_forward_vel = 0.0
            self.current_side_vel = 0.0

        # INIT
        if self.phase == FlightPhase.INIT:
            self.pub_setpoint(self.current_x, self.current_y, self.current_z, float('nan'))
            if self.offboard_counter == 20:
                self.pub_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)
                self.get_logger().info("Setting OFFBOARD mode")
            if self.offboard_counter == 25:
                self.pub_command(VehicleCommand.VEHICLE_CMD_DO_SET_HOME, p1=1.0)
                self.get_logger().info("Setting home position")
            if self.offboard_counter == 50:
                self.pub_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=1.0)
                self.get_logger().info("Sending ARM command")
            if self.armed and self.offboard_counter > 60:
                self.phase = FlightPhase.TAKEOFF
                self.get_logger().info("ARMED - Starting takeoff")
            self.offboard_counter += 1
            return

        # TAKEOFF
        if self.phase == FlightPhase.TAKEOFF:
            self.pub_setpoint(self.current_x, self.current_y, self.flight_altitude, float('nan'))
            if abs(self.current_z - self.flight_altitude) < 0.3:
                self.phase = FlightPhase.NAVIGATE
                self.prev_yaw = self.current_yaw  # Sync yaw
                self.last_waypoint_time = self.get_clock().now().nanoseconds / 1e9
                self.get_logger().info(f"Takeoff Complete. Syncing yaw to {self.current_yaw:.2f}")
            return

        # NAVIGATION
        if self.wp_index >= len(self.waypoints):
            self.pub_setpoint(self.current_x, self.current_y, self.flight_altitude, self.prev_yaw)
            return

        # Current waypoint target
        tx, ty = self.waypoints[self.wp_index]
        dx = tx - self.current_x
        dy = ty - self.current_y
        dist = math.hypot(dx, dy)

        lookahead = self.get_lookahead_target(lookahead_dist=2.5)
        if lookahead and dist > 1.5:
            lx, ly = lookahead
            dx_look = lx - self.current_x
            dy_look = ly - self.current_y
        else:
            dx_look = dx
            dy_look = dy

        dist_look = math.hypot(dx_look, dy_look)
        if dist_look > 0.1:
            dir_x = dx_look / dist_look
            dir_y = dy_look / dist_look
        else:
            dir_x = dir_y = 0.0

        yaw_target = math.atan2(dx_look, dy_look)
        yaw_alpha = 0.2
        self.prev_yaw = (1 - yaw_alpha) * self.prev_yaw + yaw_alpha * yaw_target
        yaw_goal = self.prev_yaw

        # === DYNAMIC SPEED (Balanced) ===
        min_clearance = min(self.fused_left, self.fused_right)
        
        # Reduced fear: Only slow down base speed if VERY tight
        if min_clearance < 1.5:
            base_speed = 5.5
        elif min_clearance < 3.0:
            base_speed = 7.5
        else:
            base_speed = 9.5

        # Angle gating
        heading_error = abs(math.atan2(dy_look, dx_look))
        if heading_error < math.radians(15): speed_factor = 1.0
        elif heading_error < math.radians(45): speed_factor = 0.8
        else: speed_factor = 0.5
        base_speed *= speed_factor

        # BALANCED BRAKING: Don't panic early
        # Formula: 0.4 * vel (lighter than 0.5)
        stopping_dist = 0.4 * self.current_forward_vel + 0.5
        
        # Only start braking if obstacle is within stopping distance + small buffer (1.5m)
        # Prevents slowing down for walls 5-6m away
        if self.fused_front < stopping_dist + 1.5:
            obstacle_factor = (self.fused_front - 1.2) / (stopping_dist + 0.5)
            obstacle_factor = max(0.0, min(1.0, obstacle_factor))
            base_speed *= obstacle_factor
        
        if self.fused_front < 1.2: base_speed *= 0.2
        if self.fused_front < 0.8: base_speed = 0.0

        # === ACCELERATION ===
        max_accel = 3.5
        max_vel_change = max_accel * dt
        vel_diff = base_speed - self.current_forward_vel
        if abs(vel_diff) > max_vel_change:
            vel_diff = max_vel_change if vel_diff > 0 else -max_vel_change
        self.current_forward_vel += vel_diff
        forward_speed = self.current_forward_vel

        # === BALANCED LATERAL AVOIDANCE ===
        # Use a "Deadzone" - if wall > 1.8m away, ignore it (fly straight)
        safe_zone = 1.8 
        
        repulsion = 0.0
        
        # Linear push (predictable) instead of exponential (panic)
        # Left Obstacle -> Push Right (Negative Y)
        if self.fused_left < safe_zone:
            push = (safe_zone - self.fused_left) * 1.5 # Gain 1.5
            repulsion -= push
            
        # Right Obstacle -> Push Left (Positive Y)
        if self.fused_right < safe_zone:
            push = (safe_zone - self.fused_right) * 1.5
            repulsion += push
        
        # Wall Slide: Trigger later (closer to wall)
        wall_slide_vel = 0.0
        if self.fused_front < 2.0: # Reduced from 2.5
            slide_gain = 2.0
            if self.fused_left > self.fused_right:
                wall_slide_vel = slide_gain * (1.0 - self.fused_front/2.0)
            else:
                wall_slide_vel = -slide_gain * (1.0 - self.fused_front/2.0)

        target_side_vel = repulsion + wall_slide_vel
        target_side_vel = max(min(target_side_vel, 3.5), -3.5)

        alpha_side = 0.25
        self.current_side_vel = (1 - alpha_side) * self.current_side_vel + alpha_side * target_side_vel
        lateral_speed = self.current_side_vel

        # === CRITICAL SAFETY ===
        # Reduced distance: Only panic if VERY close (< 0.6m)
        if self.fused_front < 0.6:
            forward_speed = -0.5
            # Dodge logic
            if self.fused_left > self.fused_right:
                lateral_speed = 2.0
            else:
                lateral_speed = -2.0
            self.get_logger().warn("CRITICAL: Too Close!")

        # Stuck detection
        if self.is_stuck() and dist > 1.5:
            self.stuck_counter += 1
            if self.stuck_counter > 20:
                self.wp_index += 1
                self.stuck_counter = 0
                self.publish_visual_path()
                if self.wp_index >= len(self.waypoints): self.phase = FlightPhase.REACHED
                return
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)

        # Compute global velocity
        tan_x = -dir_y
        tan_y = dir_x
        nx = self.current_x + (dir_x * forward_speed + tan_x * lateral_speed) * dt
        ny = self.current_y + (dir_y * forward_speed + tan_y * lateral_speed) * dt
        nz = self.flight_altitude

        if dist < 1.0:
            now = self.get_clock().now().nanoseconds / 1e9
            elapsed = now - self.last_waypoint_time
            self.last_waypoint_time = now
            self.wp_index += 1
            self.publish_visual_path()
            if self.wp_index >= len(self.waypoints):
                self.phase = FlightPhase.REACHED
                self.get_logger().info("✓ All waypoints reached!")
            else:
                self.get_logger().info(f"✓ WP {self.wp_index-1} ({elapsed:.1f}s) → WP {self.wp_index}")

        if self.offboard_counter % 20 == 0:
            self.get_logger().info(
                f"S:{forward_speed:.1f} Side:{lateral_speed:.1f} F:{self.fused_front:.1f} L:{self.fused_left:.1f} R:{self.fused_right:.1f}"
            )

        self.pub_setpoint(nx, ny, nz, yaw_goal)
        self.offboard_counter += 1

    def pub_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def pub_setpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position = [y, x, -z]
        msg.yaw = yaw
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
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
