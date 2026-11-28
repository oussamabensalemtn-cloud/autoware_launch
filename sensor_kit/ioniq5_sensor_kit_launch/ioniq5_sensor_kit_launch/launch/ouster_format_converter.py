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

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField

# Ouster /ouster/points layout (point_step = 48, with gaps)
IN_DTYPE = np.dtype({
    "names":   ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "ambient", "range"],
    "formats": ["<f4", "<f4", "<f4", "<f4", "<u4", "<u2", "<u2", "<u2", "<u4"],
    "offsets": [0, 4, 8, 16, 20, 24, 26, 28, 32],
    "itemsize": 48,
})

# Autoware PointXYZIRCAEDT layout (point_step = 32)
OUT_DTYPE = np.dtype({
    "names":   ["x", "y", "z", "intensity", "return_type", "channel",
                "azimuth", "elevation", "distance", "time_stamp"],
    "formats": ["<f4", "<f4", "<f4", "u1", "u1", "<u2", "<f4", "<f4", "<f4", "<u4"],
    "offsets": [0, 4, 8, 12, 13, 14, 16, 20, 24, 28],
    "itemsize": 32,
})

OUT_FIELDS = [
    PointField(name="x",           offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name="y",           offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name="z",           offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity",   offset=12, datatype=PointField.UINT8,   count=1),
    PointField(name="return_type", offset=13, datatype=PointField.UINT8,   count=1),
    PointField(name="channel",     offset=14, datatype=PointField.UINT16,  count=1),
    PointField(name="azimuth",     offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name="elevation",   offset=20, datatype=PointField.FLOAT32, count=1),
    PointField(name="distance",    offset=24, datatype=PointField.FLOAT32, count=1),
    PointField(name="time_stamp",  offset=28, datatype=PointField.UINT32,  count=1),
]


class OusterToXYZIRCAEDT(Node):
    def __init__(self):
        super().__init__("ouster_format_converter")
        self.pub = self.create_publisher(PointCloud2, "output", qos_profile_sensor_data)
        self.sub = self.create_subscription(
            PointCloud2, "input", self.on_cloud, qos_profile_sensor_data)
        # changing the frame id name from os_lidar to avoid tf issues 
        self.frame_id = self.declare_parameter("frame_id", "ouster_top").value

    def on_cloud(self, msg: PointCloud2):
        if msg.point_step != IN_DTYPE.itemsize:
            self.get_logger().warn_once(
                f"Unexpected point_step {msg.point_step}, expected {IN_DTYPE.itemsize}")
            return

        raw = np.frombuffer(msg.data, dtype=IN_DTYPE, count=msg.width * msg.height)

        out = np.empty(raw.shape[0], dtype=OUT_DTYPE)
        out["x"] = raw["x"]
        out["y"] = raw["y"]
        out["z"] = raw["z"]

        # Ouster signal is float and can exceed 255; clip into Autoware's uint8 intensity.
        intensity_raw = np.nan_to_num(raw["intensity"], nan=0.0)
        out["intensity"] = np.clip(intensity_raw, 0, 255).astype(np.uint8)

        out["return_type"] = 0                       # single-return sensor
        out["channel"] = raw["ring"].astype(np.uint16)

        x = raw["x"]; y = raw["y"]; z = raw["z"]
        # Azimuth in the scan direction: the Ouster spins clockwise (seen from above), so within a ring the columns
        # come in clockwise order and atan2 (counter-clockwise) would decrease from point to point. Autoware's
        # ring_outlier_filter requires the azimuth to increase along a ring (step < 1 deg), otherwise it drops every
        # point (older versions hid this with a degree/radian bug in that check). The distortion corrector derives
        # sign and offset of the azimuth convention from the data, so a clockwise azimuth is fine there.
        out["azimuth"] = np.mod(-np.arctan2(y, x), 2.0 * np.pi).astype(np.float32)
        out["elevation"] = np.arctan2(z, np.hypot(x, y)).astype(np.float32)
        out["distance"] = np.sqrt(x * x + y * y + z * z).astype(np.float32)

        # Ouster 't' is ns since scan start — same convention Autoware expects.
        out["time_stamp"] = raw["t"].astype(np.uint32)

        cloud = PointCloud2()
        cloud.header.stamp = msg.header.stamp
        cloud.header.frame_id = self.frame_id
        cloud.height = msg.height
        cloud.width = msg.width
        cloud.fields = OUT_FIELDS
        cloud.is_bigendian = False
        cloud.point_step = OUT_DTYPE.itemsize
        cloud.row_step = OUT_DTYPE.itemsize * msg.width
        cloud.is_dense = msg.is_dense
        cloud.data = out.tobytes()
        self.pub.publish(cloud)


def main():
    rclpy.init()
    node = OusterToXYZIRCAEDT()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()