import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import subprocess
from std_srvs.srv import Empty
import pandas as pd

# TODO: Load in a saved trajectory
# At current step see how close it is to the trajectory and give reward or reset

class TurtlebotEnvironment(gym.Env, Node):
    '''
    Turtlebot training environment in gazebo that follows the gymnasium interface so we can run PPO 
    (Proximity Policy Optimisation) to train it to follow a path
    '''
    def __init__(self, trajectory_csv: str) -> None:
        super().__init__('turtlebot_environment')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.subscription_position = self.create_subscription(
            Odometry,
            '/odom',
            self.position_callback,
            10)
        self.bridge = CvBridge()
        self.current_image = None
        self.position = None

        N_CHANNELS = 3
        HEIGHT = 256
        WIDTH = 256

        self.action_space = spaces.Discrete(3) # left, right or forward
        # Image as input, docs say channel can be first or last
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        self.current_image = None

        self.load_trajectory(trajectory_csv)

        self.reset_world_client = self.create_client(Empty, '/reset_world')
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset world service not available, waiting again...')

        self.reset()
        print("INITIALISED")
        
    def reset_world(self):
        '''
        Reset the gazebo environment, including the turtlebot position. Equivalent to calling:
        ros2 service call /reset_world std_srvs/srv/Empty
        '''
        req = Empty.Request()
        future = self.reset_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('World reset successfully')
        else:
            self.get_logger().error('Failed to reset world')

    def load_trajectory(self, trajectory_csv: str):
        '''
        Load a trajectory CSV created with the ImageRead script. This will be the reference for this model
        '''
        df = pd.read_csv(trajectory_csv)

        self.x_positions = df['x']
        self.y_positions = df['y']

    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.current_image = cv_image

    def position_callback(self, msg):
        self.position = msg.pose.pose.position

    def send_movement_command(self, vel: float, angular_vel: float):
        twist_msg = Twist()
        twist_msg.linear.x = vel
        twist_msg.angular.z = angular_vel
        self.publisher_.publish(twist_msg)

    def get_trajectory_deviation(self) -> float:
        '''
        Using the current pose of the turtlebot, return its distance from the closest point
        in the model trajectory
        '''
        if self.position is None:
            return 0
        closest_point_distance = float('inf')
        for x, y in zip(self.x_positions, self.y_positions):
            closest_point_distance = min(closest_point_distance, math.sqrt((x-self.position.x)**2+(y-self.position.y)**2))
        return closest_point_distance



    def step(self, action):
        rclpy.spin_once(self)
        # https://gymnasium.farama.org/api/env/#gymnasium.Env.step
        # Action will be the direction 0,1,2
        terminated = False
        observation = self._get_observation()
        info = self._get_info()
        truncated = False

        deviation = self.get_trajectory_deviation()
        reward = -deviation + 0.05
        if deviation > 0.1:
            # Turtlebot is too far from the path, end this episode early
            truncated = True
            terminated = True
        # print(self.get_trajectory_deviation())
        print(reward)

        angular_vel = 0
        if action == 0:
            angular_vel = -0.1
        elif action == 1:
            angular_vel = 0.0
        elif action == 2:
            angular_vel = 0.1

        self.send_movement_command(vel=0.1, angular_vel=angular_vel)

        return observation, reward, terminated, truncated, info
        

    def reset(self, seed=None):
        print("RESET")
        self.reset_world()
        self.current_image = None
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    def _get_observation(self):
        '''
        Helper function to get the current observation that is used in step and reset
        '''
        # Hard-code loading a local PNG file for now
        image_path = "images/loop/0.png"
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        
        # Convert the image to a numpy array and ensure it's in the correct format
        cv_image = cv2.resize(cv_image, (self.observation_space.shape[1], self.observation_space.shape[0]))
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if needed
        self.current_image = np.array(cv_image, dtype=np.uint8)
        return self.current_image
    
    def _get_info(self):
        '''
        Auxiliary information returned by step and reset
        '''
        return {}


    