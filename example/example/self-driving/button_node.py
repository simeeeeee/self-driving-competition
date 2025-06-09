#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2023/08/28
# ROS 2 Node for Detecting Button Press Events

# import rclpy
# from rclpy.node import Node
# from ros_robot_controller_msgs.msg import ButtonState

# class ButtonPressReceiver(Node):

#     def __init__(self, name):
#         super().__init__(name)
#         self.create_subscription(ButtonState, '/ros_robot_controller/button', self.button_callback, 10)
#         self.get_logger().info('ButtonPressReceiver node started')

#     def button_callback(self, msg):
#         if msg.id == 1:
#             self.process_button_press('Button 1', msg.state)
#         elif msg.id == 2:
#             self.process_button_press('Button 2', msg.state)

#     def process_button_press(self, button_name, state):
#         if state == 1:
#             self.get_logger().info(f'{button_name} short press detected')
#             # You can add additional logic here for short press
#         elif state == 2:
#             self.get_logger().info(f'{button_name} long press detected')
#             # You can add additional logic here for long press

# def main():
#     rclpy.init()
#     node = ButtonPressReceiver('button_press_receiver')
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
#         print('shutdown finish')

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
# encoding: utf-8
# ROS 2 Node for Detecting Button Press Events and Starting Self-Driving

import rclpy
from rclpy.node import Node
from ros_robot_controller_msgs.msg import ButtonState
from std_srvs.srv import SetBool

class ButtonPressReceiver(Node):

    def __init__(self, name):
        super().__init__(name)
        self.create_subscription(ButtonState, '/ros_robot_controller/button', self.button_callback, 10)
        self.cli = self.create_client(SetBool, '/self_driving/set_running')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /self_driving/set_running service...')

        self.get_logger().info('ButtonPressReceiver node started')

    def button_callback(self, msg):
        if msg.id == 1:
            self.process_button_press('Button 1', msg.state)

    def process_button_press(self, button_name, state):
        if state == 1:  # Short press
            self.get_logger().info(f'{button_name} short press detected')
            self.call_self_driving_service(True)

    def call_self_driving_service(self, run: bool):
        req = SetBool.Request()
        req.data = run
        future = self.cli.call_async(req)

        def callback(fut):
            try:
                res = fut.result()
                if res.success:
                    self.get_logger().info(f"Self-driving started: {res.message}")
                else:
                    self.get_logger().warn(f"Failed to start self-driving: {res.message}")
            except Exception as e:
                self.get_logger().error(f"Service call failed: {str(e)}")

        future.add_done_callback(callback)

def main():
    rclpy.init()
    node = ButtonPressReceiver('button_press_receiver')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print('shutdown finish')

if __name__ == '__main__':
    main()
