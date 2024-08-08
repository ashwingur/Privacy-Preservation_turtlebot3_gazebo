import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

'''
This is just a test script I made to see if it correctly resets the turtlebot environment. 
Note the gazebo world has to be launched first.
'''

class ResetWorldClient(Node):
    def __init__(self):
        super().__init__('reset_world_client')
        self.client = self.create_client(Empty, '/reset_world')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = Empty.Request()

    def send_request(self):
        self.future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    reset_world_client = ResetWorldClient()
    response = reset_world_client.send_request()
    reset_world_client.get_logger().info('World reset successfully')
    reset_world_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
