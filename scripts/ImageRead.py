import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
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
        
        self.subscription_position = self.create_subscription(
            Odometry,
            '/odom',
            self.position_callback,
            10)

        self.bridge = CvBridge()
        self.cv_image = None
        self.velocity = Twist()
        self.position = None
        self.last_image_time = time.time()
        self.data = []

        # How often an image will be capture in seconds
        self.IMAGE_FREQUENCY = 0.5

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error('Error converting ROS Image to OpenCV image: %s' % str(e))
            return

        # if IMAGE_FREQUENCY seconds have passed since the last capture, then do another capture
        current_time = time.time()
        if current_time - self.last_image_time >= self.IMAGE_FREQUENCY:
            self.last_image_time = current_time
            print(self.last_image_time)
            self.save_data()

        # Display the image
        # cv2.imshow('Camera Image', self.cv_image)
        # cv2.waitKey(0)

    def velocity_callback(self, msg):
        self.velocity = msg

    def position_callback(self, msg):
        self.position = msg.pose.pose.position

    def save_data(self):
        if self.cv_image is not None:
            data_entry = {'image': self.cv_image, 'velocity': self.velocity.linear.x, 'angular_velocity': self.velocity.angular.z}
            if self.position is not None:
                data_entry['x'] = self.position.x
                data_entry['y'] = self.position.y
            else:
                data_entry['x'] = 0.0
                data_entry['y'] = 0.0

            self.data.append(data_entry)
            print(f'Saved datapoint: {len(self.data) - 1}')

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    try:
        while rclpy.ok():
            rclpy.spin_once(image_subscriber)
            # After x images, break (extend later to allow for a keypress to break)
            # if len(image_subscriber.data) == 3:
            #     save_to_file(image_subscriber.data)
            #     break
    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPT, Saving data')
        save_to_file(image_subscriber.data)

    # Clean up
    image_subscriber.destroy_node()
    rclpy.shutdown()

def save_to_file(data):
    with open('image_velocity_data.csv', mode='w', newline='') as file:
        IMAGE_PATH = 'images'
        if not os.path.exists(IMAGE_PATH):
            os.makedirs(IMAGE_PATH)
        
        # Delete all old data
        for filename in os.listdir(IMAGE_PATH):
            if filename.endswith('.png'):
                os.remove(os.path.join(IMAGE_PATH, filename))

        writer = csv.DictWriter(file, fieldnames=['image', 'velocity', 'angular_velocity', 'x', 'y'])
        writer.writeheader()
        for index, entry in enumerate(data):
            image_name = f'images/camera_image_{index}.png'
            writer.writerow({'image': image_name, 'velocity': entry['velocity'], 'angular_velocity': entry['angular_velocity'], 'x': entry['x'],'y': entry['y']})
            cv2.imwrite(image_name, entry['image'])
    
    print("Data saved to image_velocity_data.csv")

if __name__ == '__main__':
    main()
