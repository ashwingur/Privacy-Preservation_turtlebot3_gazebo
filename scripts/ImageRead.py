import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import time
import csv
import os

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription_image = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.subscription_velocity = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.velocity_callback,
            10)
        # self.subscription_image  # prevent unused variable warning
        # self.subscription_velocity  # prevent unused variable warning
        self.bridge = CvBridge()
        self.cv_image = None
        self.velocity = Twist()
        self.last_image_time = time.time()
        self.data = []

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error('Error converting ROS Image to OpenCV image: %s' % str(e))
            return

        # Check if 1.2 seconds have passed since the last image capture
        current_time = time.time()
        if current_time - self.last_image_time >= 1.2:
            self.last_image_time = current_time
            self.save_data()

        # Display the image
        # cv2.imshow('Camera Image', self.cv_image)
        # cv2.waitKey(0)

    def velocity_callback(self, msg):
        self.velocity = msg

    def save_data(self):
        if self.cv_image is not None:
            data_entry = {'image': self.cv_image, 'velocity': self.velocity.linear.x, 'angular_velocity': self.velocity.angular.z}
            self.data.append(data_entry)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    try:
        while rclpy.ok():
            rclpy.spin_once(image_subscriber)
            # After 10 images, break (extend later)
            if len(image_subscriber.data) > 10:
                save_to_file(image_subscriber.data)
                break
    except KeyboardInterrupt:
        pass

    # Clean up
    image_subscriber.destroy_node()
    rclpy.shutdown()

def save_to_file(data):
    with open('image_velocity_data.csv', mode='w', newline='') as file:
        if not os.path.exists('images'):
            os.makedirs('images')

        writer = csv.DictWriter(file, fieldnames=['image', 'velocity', 'angular_velocity'])
        writer.writeheader()
        for index, entry in enumerate(data):
            image_name = f'images/camera_image_{index}.png'
            writer.writerow({'image': image_name, 'velocity': entry['velocity'], 'angular_velocity': entry['angular_velocity']})
            cv2.imwrite(image_name, entry['image'])
    
    print("Data saved to image_velocity_data.csv")

if __name__ == '__main__':
    main()
