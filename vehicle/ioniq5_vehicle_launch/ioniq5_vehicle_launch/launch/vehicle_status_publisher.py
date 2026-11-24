#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

# Import all necessary Autoware vehicle messages
from autoware_vehicle_msgs.msg import (
    SteeringReport,
    GearReport,
    ControlModeReport,
    TurnIndicatorsCommand,
    TurnIndicatorsReport,
    HazardLightsCommand,
    HazardLightsReport
)

class CustomVehicleInterfaceNode(Node):
    def __init__(self):
        super().__init__('custom_vehicle_interface')

        # ---------------- Publishers ----------------
        self.steer_pub = self.create_publisher(
            SteeringReport, '/vehicle/status/steering_status', 10)
        self.gear_pub = self.create_publisher(
            GearReport, '/vehicle/status/gear_status', 10)
        self.control_mode_pub = self.create_publisher(
            ControlModeReport, '/vehicle/status/control_mode', 10)
        self.turn_pub = self.create_publisher(
            TurnIndicatorsReport, '/vehicle/status/turn_indicators_status', 10)
        self.hazard_pub = self.create_publisher(
            HazardLightsReport, '/vehicle/status/hazard_lights_status', 10)

        # ---------------- Subscribers ----------------
        self.create_subscription(
            Float64, '/steering_angle', self.steering_callback, 10)
        
        self.create_subscription(
            TurnIndicatorsCommand, '/planning/turn_indicators_cmd', self.turn_callback, 10)
            
        self.create_subscription(
            HazardLightsCommand, '/planning/hazard_lights_cmd', self.hazard_callback, 10)

        # Timer to publish constant states (Gear & Control Mode) at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("Custom Vehicle Interface Node Started. Publishing constants and relaying commands.")

    # 1. Relaying Steering Angle
    def steering_callback(self, msg: Float64):
        report = SteeringReport()
        report.stamp = self.get_clock().now().to_msg()
        # Convert the incoming Float64 data to the expected float32 for steering_tire_angle
        report.steering_tire_angle = float(msg.data)
        self.steer_pub.publish(report)

    # 2. Relaying Turn Indicators (Command -> Report)
    def turn_callback(self, msg: TurnIndicatorsCommand):
        report = TurnIndicatorsReport()
        report.stamp = self.get_clock().now().to_msg()
        # The command msg uses the field 'command', the report msg uses 'report'
        report.report = msg.command
        self.turn_pub.publish(report)

    # 3. Relaying Hazard Lights (Command -> Report)
    def hazard_callback(self, msg: HazardLightsCommand):
        report = HazardLightsReport()
        report.stamp = self.get_clock().now().to_msg()
        # The command msg uses the field 'command', the report msg uses 'report'
        report.report = msg.command
        self.hazard_pub.publish(report)

    # 4. Publishing Constant States
    def timer_callback(self):
        now = self.get_clock().now().to_msg()

        # Publish Gear Status as DRIVE
        gear_msg = GearReport()
        gear_msg.stamp = now
        gear_msg.report = GearReport.DRIVE
        self.gear_pub.publish(gear_msg)

        # Publish Control Mode as AUTONOMOUS
        mode_msg = ControlModeReport()
        mode_msg.stamp = now
        mode_msg.mode = ControlModeReport.AUTONOMOUS
        self.control_mode_pub.publish(mode_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CustomVehicleInterfaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()