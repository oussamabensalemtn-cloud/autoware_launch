#!/usr/bin/env python3
# Copyright 2026 Tobias Schulz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo


class CameraInfoRelay(Node):
    def __init__(self):
        super().__init__('camera_info_relay')

        self.declare_parameter('input_topic', '')
        self.declare_parameter('output_topic', '')
        self.declare_parameter('distortion_model', 'plumb_bob')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.distortion_model = (
            self.get_parameter('distortion_model').get_parameter_value().string_value
        )

        if not input_topic or not output_topic:
            raise ValueError('Both input_topic and output_topic parameters must be set.')

        # Subscribe BEST_EFFORT (compatible with both RELIABLE and BEST_EFFORT drivers)
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publish RELIABLE (compatible with both RELIABLE and BEST_EFFORT subscribers)
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(CameraInfo, output_topic, pub_qos)
        self.sub = self.create_subscription(CameraInfo, input_topic, self.callback, sub_qos)

    def callback(self, msg: CameraInfo):
        msg.distortion_model = self.distortion_model
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Ensure P is populated from K if any camera publishes a zeroed projection matrix
        if msg.p[0] == 0.0 and msg.k[0] != 0.0:
            msg.p = [
                msg.k[0], msg.k[1], msg.k[2], 0.0,
                msg.k[3], msg.k[4], msg.k[5], 0.0,
                msg.k[6], msg.k[7], msg.k[8], 0.0,
            ]

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CameraInfoRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()