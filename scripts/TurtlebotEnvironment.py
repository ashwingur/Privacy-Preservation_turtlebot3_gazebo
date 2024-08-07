import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

class TurtlebotEnvironment(gym.Env, Node):
    def __init__(self) -> None:
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

        N_CHANNELS = 3
        HEIGHT = 256
        WIDTH = 256

        self.action_space = spaces.Discrete(3) # left, right or forward
        # Image as input, docs say channel can be first or last
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        self.current_image = None

        self.reset()

    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.current_image = cv_image

    def position_callback(self, msg):
        self.position = msg.pose.pose.position


    def step(self, action):
        # https://gymnasium.farama.org/api/env/#gymnasium.Env.step
        # Action will be the direction 0,1,2
        terminated = False
        reward = 1 if terminated else 0
        observation = self._get_observation()
        info = self._get_info()
        truncated = False

        return observation, reward, terminated, truncated, info
        

    def reset(self, seed=None):
        print("RESET")
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


    