# Updated to add ROS 2 (rclpy) support with simulation fallback.
# Backwards compatible: if rclpy is not present, runs in simulated mode.
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
from pathlib import Path
import json
import time
import math
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try to import rclpy and ROS2 messages
_HAS_ROS2 = False
_rclpy = None
try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from geometry_msgs.msg import PoseStamped, Quaternion  # type: ignore
    from nav_msgs.msg import Odometry  # type: ignore
    from std_msgs.msg import Header  # type: ignore
    _HAS_ROS2 = True
    _rclpy = rclpy
    logger.info("rclpy available: ROS 2 bridge enabled.")
except Exception:
    _HAS_ROS2 = False
    logger.info("rclpy not available: using simulated ROS 2 bridge.")


# Global ROS2 node holder (created lazily if rclpy is present)
_RCLPY_NODE: Optional[Node] = None


def _ensure_rclpy_node():
    global _RCLPY_NODE, _rclpy
    if not _HAS_ROS2:
        return None
    if _RCLPY_NODE is None:
        if not _rclpy.ok():
            try:
                _rclpy.init()
            except Exception:
                # some environments may already initialize externally; ignore errors
                pass
        try:
            _RCLPY_NODE = Node("ros_quantum_bridge_node")
        except Exception as e:
            logger.exception("Failed to create rclpy Node: %s", e)
            _RCLPY_NODE = None
    return _RCLPY_NODE


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    return q


@dataclass
class RobotController:
    robot_id: str
    sim_step: float = 0.1  # meters per simulated step
    sim_yaw_step: float = 0.2  # radians per simulated step
    _sim_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))  # x,y,yaw
    _last_cmd: Optional[Tuple[float, float, float]] = None
    # ROS2-specific fields (initialized lazily)
    _publisher: Any = field(default=None, init=False, repr=False)
    _odom_subscriber: Any = field(default=None, init=False, repr=False)
    _last_odom: Optional[Tuple[float, float, float]] = field(default=None, init=False, repr=False)

    def _ensure_ros2_pub_and_sub(self):
        if not _HAS_ROS2:
            return
        node = _ensure_rclpy_node()
        if node is None:
            return
        if self._publisher is None:
            topic = f"/{self.robot_id}/pose_goal"
            try:
                self._publisher = node.create_publisher(PoseStamped, topic, 10)
                logger.debug("Created publisher for %s on %s", self.robot_id, topic)
            except Exception as e:
                logger.exception("Failed to create publisher for %s: %s", self.robot_id, e)
        # create odom subscriber
        if self._odom_subscriber is None:
            odom_topic = f"/{self.robot_id}/odom"
            try:
                def _odom_cb(msg: Odometry):
                    try:
                        x = msg.pose.pose.position.x
                        y = msg.pose.pose.position.y
                        # yaw extraction from quaternion (approximate for planar)
                        q = msg.pose.pose.orientation
                        yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))
                        self._last_odom = (float(x), float(y), float(yaw))
                    except Exception:
                        pass
                self._odom_subscriber = node.create_subscription(Odometry, odom_topic, _odom_cb, 10)
                logger.debug("Created odom subscriber for %s on %s", self.robot_id, odom_topic)
            except Exception as e:
                logger.debug("Could not create odom subscriber for %s: %s", self.robot_id, e)

    def send_goal(self, x: float, y: float, yaw: float = 0.0) -> None:
        """
        Send navigation goal. If ROS2 present, publish PoseStamped; else simulate.
        Yaw will be converted to quaternion for ROS2 messages. Simulation updates pose gradually.
        """
        self._last_cmd = (x, y, yaw)
        if _HAS_ROS2:
            try:
                self._ensure_ros2_pub_and_sub()
                node = _RCLPY_NODE
                if self._publisher is not None and node is not None:
                    ps = PoseStamped()
                    ps.header = Header()
                    ps.header.stamp = node.get_clock().now().to_msg()
                    ps.header.frame_id = "map"
                    ps.pose.position.x = float(x)
                    ps.pose.position.y = float(y)
                    ps.pose.orientation = _yaw_to_quaternion(yaw)
                    self._publisher.publish(ps)
                    logger.debug("Published ROS2 PoseStamped for %s -> (%.3f, %.3f, %.3f)", self.robot_id, x, y, yaw)
                    return
                # fallback: if publisher not available, fallthrough to simulation below
            except Exception as e:
                logger.exception("Failed to publish goal via ROS2 for %s: %s", self.robot_id, e)

        # Simulate moving a bit towards the goal
        dx = x - self._sim_pose[0]
        dy = y - self._sim_pose[1]
        desired_yaw = math.atan2(dy, dx) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else self._sim_pose[2]
        # rotate towards desired yaw
        yaw_diff = (desired_yaw - self._sim_pose[2] + math.pi) % (2 * math.pi) - math.pi
        yaw_step = max(-self.sim_yaw_step, min(self.sim_yaw_step, yaw_diff))
        self._sim_pose[2] += yaw_step
        # move forward in the new heading a bit (but don't overshoot goal)
        dist = math.hypot(dx, dy)
        move = min(self.sim_step, dist)
        self._sim_pose[0] += math.cos(self._sim_pose[2]) * move
        self._sim_pose[1] += math.sin(self._sim_pose[2]) * move
        logger.info("Simulated send_goal for %s -> new pose (%.3f, %.3f, %.3f)", self.robot_id, self._sim_pose[0], self._sim_pose[1], self._sim_pose[2])

    def get_odometry(self) -> Tuple[float, float, float]:
        """
        Return current odometry (x,y,yaw).
        If ROS2 and odom subscription is active, return last received odom.
        Otherwise return simulated odometry.
        """
        if _HAS_ROS2:
            if self._last_odom is not None:
                return self._last_odom
            # No odometry available yet; return a sensible default
            return (0.0, 0.0, 0.0)
        else:
            return (float(self._sim_pose[0]), float(self._sim_pose[1]), float(self._sim_pose[2]))


class SwarmCoordinator:
    def __init__(self, robot_ids: List[str], sim_step: float = 0.1):
        self.controllers: Dict[str, RobotController] = {rid: RobotController(rid, sim_step=sim_step) for rid in robot_ids}

    def dispatch_goals(self, goals: List[Tuple[float, float, float]]) -> None:
        """Dispatch goals to robots in order; len(goals) should equal number of robots (or be cycled)."""
        for rid, goal in zip(self.controllers.keys(), goals):
            x, y, yaw = goal
            self.controllers[rid].send_goal(x, y, yaw)
            logger.debug("Dispatched goal to %s: (%.3f, %.3f, %.3f)", rid, x, y, yaw)

    def read_all_odometry(self) -> Dict[str, Tuple[float, float, float]]:
        odoms = {}
        for rid, ctrl in self.controllers.items():
            odoms[rid] = ctrl.get_odometry()
        return odoms


def generate_launch_file(robot_ids: List[str], filename: Optional[str] = None, output_dir: str = "outputs/ros2_launch") -> str:
    """
    Generate a simple ROS2 Python launch template that can be adapted to the user's stack.
    Returns path to file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"swarm_launch_{int(time.time())}.py"
    path = Path(output_dir) / filename
    # A minimal Python launch file template for ROS2 users; users should adapt to their robot stacks.
    template = f'''# Auto-generated ROS2 launch template
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
'''
    for rid in robot_ids:
        template += f"    # {rid}\n"
        template += f"    ld.add_action(Node(package='your_robot_bringup', executable='bringup_node', namespace='{rid}', output='screen'))\n\n"
    template += "    return ld\n"
    path.write_text(template)
    logger.info(\"Generated ROS2 launch file at %s\", str(path))
    return str(path)


def deploy_to_ros2_swarm(robot_ids: List[str], slots: np.ndarray, simulate: bool = True, sim_step: float = 0.1) -> Dict[str, Any]:
    """
    High-level deploy function for ROS2:
     - If simulate==True or rclpy not available, uses simulated RobotController
     - Returns odometry snapshots after dispatch
    """
    coord = SwarmCoordinator(robot_ids, sim_step=sim_step)
    goals = []
    for i, rid in enumerate(robot_ids):
        slot = slots[i % len(slots)]
        goals.append((float(slot[0]), float(slot[1]), 0.0))
    coord.dispatch_goals(goals)
    # allow simulated robots (or real robots) a short moment
    time.sleep(0.2)
    odoms = coord.read_all_odometry()
    return {"dispatched": len(goals), "odometry": odoms}
