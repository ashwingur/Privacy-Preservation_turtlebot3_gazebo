'''Using the trained model trajectory this script will capture live images and
evaluate where it is in the trajectory where it will then send a steering command
to control the turtlebot to follow the trajectory as closely as possible.
'''

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import threading
import cv2
import torch
import numpy as np
from TrajectoryTrain import TurtlebotCNN
import matplotlib.pyplot as plt
import pandas as pd

class TurtlebotDrive(Node):
    def __init__(self):
        super().__init__('command_turtlebot')
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

        # Initialise velocity and angular velocity
        # We always start by going straight
        self.velocity = 0.1
        self.ang_velocity = 0

        # Keep track of x,y positions for reporting
        self.position = None
        self.x_positions = []
        self.y_positions = []
        
        # Load the machine learning model
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TurtlebotCNN().to(self.device)
        self.model.load_state_dict(torch.load('model_weights.pth'))
        self.model.eval()
        self.IMAGE_RESIZE_W = 384
        self.IMAGE_RESIZE_H = 216

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.current_image = cv_image

        # cv2.imshow("Image from TurtleBot Camera", cv_image)
        # cv2.waitKey(1)  # This line is necessary to update the imshow window

    def position_callback(self, msg):
        self.position = msg.pose.pose.position

    def record_position(self):
        self.x_positions.append(self.position.x)
        self.y_positions.append(self.position.y)

    def send_movement_command(self, vel: float, angular_vel: float):
        twist_msg = Twist()
        twist_msg.linear.x = vel
        twist_msg.angular.z = angular_vel
        self.publisher_.publish(twist_msg)

    def control_loop(self):
        if self.current_image is None:
            return
        # Process the current image. Resize and convert to tensor first
        target_size = (self.IMAGE_RESIZE_W, self.IMAGE_RESIZE_H)
        image = cv2.resize(self.current_image, target_size)
        
        # Transpose from RGB to tensor format and add a batch dimension to the start
        image_tensor = torch.from_numpy(np.expand_dims(np.transpose(image, (2, 1, 0)), axis=0)).float()
        image_tensor = image_tensor.to(self.device)
        
        # Normalize tensor between 0 and 1
        min_value = torch.min(image_tensor)
        max_value = torch.max(image_tensor)
        image_tensor = (image_tensor - min_value) / (max_value - min_value)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        label = predicted.item()
        # Now that we have a prediction, update the steering
        # The label values are determined in the TurtlebotDataLoader.py
        if label == 0:
            # Right
            self.ang_velocity = -0.2
            print('right')
        elif label == 1:
            # Straight
            self.ang_velocity = 0.0
            print('straight')
        elif label == 2:
            # Left
            self.ang_velocity = 0.2
            print('left')
        
        self.send_movement_command(self.velocity, self.ang_velocity)


    def plot_trajectory(self, training_csv):
        '''
        Given the training CSV generated from ImageRead.py, plots the
        trained and live trajectory for comparison purposes. Also saves the plot
        to trajectory.png
        '''
        df = pd.read_csv(training_csv)
        x_ref = df['x'].values
        y_ref = df['y'].values


        plt.plot(self.x_positions, self.y_positions, 'r-', label='Live', alpha=0.5, linewidth=2.5)  # Plot the first set in red with lines connecting consecutive points
        plt.plot(x_ref, y_ref, 'b-', label='Trained', alpha=0.5)  # Plot the second set in blue with lines connecting consecutive points
        # plt.scatter(x1, y1, color='red')  # Scatter plot the points of the first set in red
        # plt.scatter(x2, y2, color='blue')  # Scatter plot the points of the second set in blue
        plt.xlabel('X Position')  # Label for x-axis
        plt.ylabel('Y Position')  # Label for y-axis
        plt.title("Live and Trained Trajectories")
        plt.legend()  # Show legend
        plt.grid(True)  # Show grid
        plt.savefig("trajectory_comparison.png")
        plt.show()  # Show the plot



def main(args=None):
    rclpy.init(args=args)
    turtlebot = TurtlebotDrive()
    # Spin in a separate thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(turtlebot, ), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            turtlebot.control_loop()
            time.sleep(1)
            # Record current position
            turtlebot.record_position()
    except KeyboardInterrupt:
        pass
    print("Shutting down")
    spin_thread.join()
    turtlebot.destroy_node()
    turtlebot.plot_trajectory('training.csv')
    # rclpy.shutdown()

if __name__ == '__main__':
    main()

