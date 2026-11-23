#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from autoware_sensing_msgs.msg import GnssInsOrientationStamped

S = math.sqrt(0.5)
Q_NED_TO_ENU = (S, S, 0.0, 0.0)  # (x, y, z, w)
Q_FLU_TO_FRD = (1.0, 0.0, 0.0, 0.0)


def qmul(a, b):
    return (
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    )


def ned_frd_to_enu_flu(q):
    return qmul(qmul(Q_NED_TO_ENU, q), Q_FLU_TO_FRD)


class NedFrdToGnssInsOrientation(Node):
    def __init__(self):
        super().__init__('ned_frd_to_gnss_ins_orientation')

        input_topic = self.declare_parameter('input_topic', '/hi5/tf/center_rear_axle_ned').value
        output_topic = self.declare_parameter('output_topic', '/sensing/gnss/autoware_orientation').value
        self.child_frame = self.declare_parameter('child_frame_id', 'base_link_ned').value
        self.output_frame = self.declare_parameter('output_frame_id', 'base_link').value
        self.rmse_x = self.declare_parameter('rmse_rotation_x', 0.0087).value
        self.rmse_y = self.declare_parameter('rmse_rotation_y', 0.0087).value
        self.rmse_z = self.declare_parameter('rmse_rotation_z', 0.0087).value

        self.pub = self.create_publisher(GnssInsOrientationStamped, output_topic, 10)
        self.create_subscription(TFMessage, input_topic, self.on_tf, 10)

    def on_tf(self, msg: TFMessage):
        for t in msg.transforms:
            if t.child_frame_id != self.child_frame:
                continue

            r = t.transform.rotation
            qx, qy, qz, qw = ned_frd_to_enu_flu((r.x, r.y, r.z, r.w))

            out = GnssInsOrientationStamped()
            out.header.stamp = t.header.stamp
            out.header.frame_id = self.output_frame
            out.orientation.orientation.x = qx
            out.orientation.orientation.y = qy
            out.orientation.orientation.z = qz
            out.orientation.orientation.w = qw
            # must be > 0, otherwise the EKF gets zero covariance
            out.orientation.rmse_rotation_x = self.rmse_x
            out.orientation.rmse_rotation_y = self.rmse_y
            out.orientation.rmse_rotation_z = self.rmse_z

            self.pub.publish(out)
            return


def main(args=None):
    rclpy.init(args=args)
    node = NedFrdToGnssInsOrientation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()