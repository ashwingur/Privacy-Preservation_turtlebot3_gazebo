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
import sys

class ImageSubscriber(Node):

    def __init__(self, image_folder, last_image_index):
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

        self.previous_train_index = last_image_index

        # How often an image will be capture in seconds
        self.IMAGE_FREQUENCY = 0.25

        # Initialise image directory if it doesn't exist
        self.IMAGE_PATH = image_folder
        print(f'image path is {self.IMAGE_PATH}')
        if not os.path.exists(self.IMAGE_PATH):
            os.makedirs(self.IMAGE_PATH)
            print("making new folder")
        
        # Delete all old image data if present
        # for filename in os.listdir(IMAGE_PATH):
        #     if filename.endswith('.png'):
        #         os.remove(os.path.join(IMAGE_PATH, filename))

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
            # print(self.last_image_time)
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
            # data_entry = {'image': self.cv_image, 'velocity': self.velocity.linear.x, 'angular_velocity': self.velocity.angular.z}
            data_entry = {'velocity': self.velocity.linear.x, 'angular_velocity': self.velocity.angular.z}
            if self.position is not None:
                data_entry['x'] = self.position.x
                data_entry['y'] = self.position.y
            else:
                data_entry['x'] = 0.0
                data_entry['y'] = 0.0

            self.data.append(data_entry)
            print(f'Saved datapoint: {self.previous_train_index + len(self.data) - 1}')

            # Also save the image now to save some time instead of at the end
            image_name = f'{self.IMAGE_PATH}/{self.previous_train_index +len(self.data)-1}.png'
            cv2.imwrite(image_name, self.cv_image)
    



def main(args=None):
    if len(sys.argv) != 3:
        print('Usage python3 ImageRead.py <image_folder> <training csv>')
        return
    
    image_folder = sys.argv[1]
    training_csv = sys.argv[2]
    

    rclpy.init(args=args)
    image_subscriber = ImageSubscriber(image_folder, last_image_index(training_csv))
    try:
        while rclpy.ok():
            rclpy.spin_once(image_subscriber)
    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPT, Saving data')
        save_to_file(image_subscriber.data, training_csv)

    # Clean up
    image_subscriber.destroy_node()
    rclpy.shutdown()

def last_image_index(csvfile):
    if os.path.exists(csvfile):
            # Get the last index from the existing file
            with open(csvfile, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row:
                    last_index = int(last_row['image'].split('.')[0]) + 1
                else:
                    last_index = 0
    else:
        last_index = 0
    
    return last_index


def save_to_file(data, csvfile):
    file_exists = os.path.exists(csvfile)

    with open(csvfile, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['image', 'velocity', 'angular_velocity', 'x', 'y'])

        if not file_exists:
            writer.writeheader()

        if file_exists:
            # Get the last index from the existing file
            with open(csvfile, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row:
                    last_index = int(last_row['image'].split('.')[0]) + 1
                else:
                    last_index = 0
        else:
            last_index = 0

        for entry in data:
            image_name = f'{last_index}.png'
            writer.writerow({'image': image_name, 'velocity': entry['velocity'], 'angular_velocity': entry['angular_velocity'], 'x': entry['x'],'y': entry['y']})
            last_index += 1

if __name__ == '__main__':
    main()
