'''Using the trained model trajectory this script will capture live images and
evaluate where it is in the trajectory where it will then send a steering command
to control the turtlebot to follow the trajectory as closely as possible.
'''

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import cv2
import torch
import numpy as np
from TrajectoryTrain import CNN

class TurtlebotDrive(Node):
    def __init__(self):
        super().__init__('command_turtlebot')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.current_image = None

        # Initialise velocity and angular velocity
        # We always start by going straight
        self.velocity = 1
        self.ang_velocity = 0
        
        # Load the machine learning model
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN().to(self.device)
        self.model.load_state_dict(torch.load('model_weights.pth'))
        self.model.eval()
        self.IMAGE_RESIZE_W = 384
        self.IMAGE_RESIZE_H = 216

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.current_image = cv_image

        # cv2.imshow("Image from TurtleBot Camera", cv_image)
        # cv2.waitKey(1)  # This line is necessary to update the imshow window

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
        # Normalize the tensor
        # Normalize tensor between 0 and 1
        min_value = torch.min(image_tensor)
        max_value = torch.max(image_tensor)
        image_tensor = (image_tensor - min_value) / (max_value - min_value)
        # print(image_tensor)
        # Should be [3, 384, 216]
        print(image_tensor.size())
        with torch.no_grad():
            outputs = self.model(image_tensor)
            print(outputs)
            _, predicted = torch.max(outputs, 1)
            print(f'prediction is {predicted}')

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
            print("loop")
    except KeyboardInterrupt:
        pass

    turtlebot.destroy_node()
    rclpy.shutdown()
    spin_thread.join

if __name__ == '__main__':
    main()

