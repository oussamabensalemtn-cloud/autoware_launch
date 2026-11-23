#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistWithCovarianceStamped
from autoware_vehicle_msgs.msg import VelocityReport


class TwistToVelocityReport(Node):
    def __init__(self):
        super().__init__("twist_to_velocity_report")

        self.declare_parameter("input_topic", "/fzi/twist_with_covariance_stamped")
        self.declare_parameter("output_topic", "/vehicle/status/velocity_status")

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        self.pub = self.create_publisher(VelocityReport, output_topic, 10)
        self.sub = self.create_subscription(
            TwistWithCovarianceStamped,
            input_topic,
            self.callback,
            10,
        )

        self.get_logger().info(f"Converting {input_topic} -> {output_topic}")

    def callback(self, msg: TwistWithCovarianceStamped):
        out = VelocityReport()
        out.header = msg.header

        # Autoware conventions:
        # longitudinal_velocity: +forward (base_link x)
        # lateral_velocity: +left (base_link y)
        # heading_rate: +CCW yaw rate (base_link z)
        out.longitudinal_velocity = float(msg.twist.twist.linear.x)
        out.lateral_velocity = float(msg.twist.twist.linear.y)
        out.heading_rate = float(msg.twist.twist.angular.z)

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TwistToVelocityReport()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()