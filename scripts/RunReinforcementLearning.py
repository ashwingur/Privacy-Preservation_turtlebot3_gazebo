from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from TurtlebotEnvironment import TurtlebotEnvironment
import rclpy
from TrajectoryTrain import TurtlebotCNN

rclpy.init()
print("initialised rclpy")
env = TurtlebotEnvironment()

check_env(env)

# Create a new environment wrapped in a vectorized environment
env = make_vec_env(lambda: env, n_envs=1)

# Create the PPO model with the custom policy
model = PPO('CnnPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("models/ppo_turtlebot")
