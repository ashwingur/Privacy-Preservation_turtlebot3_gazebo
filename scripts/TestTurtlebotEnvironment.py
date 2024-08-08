from stable_baselines3 import PPO
from TurtlebotEnvironment import TurtlebotEnvironment
import rclpy
from stable_baselines3.common.env_util import make_vec_env

# Load the trained model
model = PPO.load("models/ppo_turtlebot")

# Initialize rclpy for ROS2 communication
rclpy.init()

# Create the environment
env = TurtlebotEnvironment(trajectory_csv='csv/left_turn.csv')

vec_env = make_vec_env(lambda: env, n_envs=1)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
