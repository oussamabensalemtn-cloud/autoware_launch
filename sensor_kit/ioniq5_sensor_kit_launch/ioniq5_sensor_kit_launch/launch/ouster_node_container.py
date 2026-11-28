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

import os
import yaml
from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterFile


def get_vehicle_info(context):
    gp = context.launch_configurations.get("ros_params", {})
    if not gp:
        gp = dict(context.launch_configurations.get("global_params", {}))
    p = {}
    p["min_longitudinal_offset"] = -gp.get("rear_overhang", 1.0)
    p["max_longitudinal_offset"] = gp.get("front_overhang", 1.0) + gp.get("wheel_base", 2.7)
    p["min_lateral_offset"] = -(gp.get("wheel_tread", 1.6) / 2.0 + gp.get("right_overhang", 0.2))
    p["max_lateral_offset"] = gp.get("wheel_tread", 1.6) / 2.0 + gp.get("left_overhang", 0.2)
    p["min_height_offset"] = 0.0
    p["max_height_offset"] = gp.get("vehicle_height", 2.0)
    return p


def get_vehicle_mirror_info(context):
    path = LaunchConfiguration("vehicle_mirror_param_file").perform(context)
    with open(path, "r") as f:
        p = yaml.safe_load(f)["/**"]["ros__parameters"]
    return p


def launch_setup(context, *args, **kwargs):
    def create_parameter_dict(*args):
        return {x: LaunchConfiguration(x) for x in args}

    distortion_corrector_node_param = ParameterFile(
        param_file=LaunchConfiguration("distortion_correction_node_param_path").perform(context),
        allow_substs=True,
    )
    ring_outlier_filter_node_param = ParameterFile(
        param_file=LaunchConfiguration("ring_outlier_filter_node_param_path").perform(context),
        allow_substs=True,
    )

    nodes = []

    # 1. Self-Crop Box Filter
    cropbox_parameters = create_parameter_dict("input_frame", "output_frame")
    cropbox_parameters["negative"] = True
    cropbox_parameters["processing_time_threshold_sec"] = 0.1  #0,01
    vehicle_info = get_vehicle_info(context)
    cropbox_parameters.update({
        "min_x": vehicle_info["min_longitudinal_offset"],
        "max_x": vehicle_info["max_longitudinal_offset"],
        "min_y": vehicle_info["min_lateral_offset"],
        "max_y": vehicle_info["max_lateral_offset"],
        "min_z": vehicle_info["min_height_offset"],
        "max_z": vehicle_info["max_height_offset"],
    })

    nodes.append(
        ComposableNode(
            package="autoware_pointcloud_preprocessor",
            plugin="autoware::pointcloud_preprocessor::CropBoxFilterComponent",
            name="crop_box_filter_self",
            remappings=[
                ("input", "/sensing/lidar/ouster_converted"),
                ("output", "self_cropped/pointcloud"),
            ],
            parameters=[cropbox_parameters],
            extra_arguments=[{"use_intra_process_comms": LaunchConfiguration("use_intra_process")}],
        )
    )

    # 2. Mirror-Crop Box Filter
    mirror_cropbox_parameters = create_parameter_dict("input_frame", "output_frame")
    mirror_cropbox_parameters["negative"] = True
    mirror_cropbox_parameters["processing_time_threshold_sec"] = 0.1  #0,01
    mirror_info = get_vehicle_mirror_info(context)
    mirror_cropbox_parameters.update({
        "min_x": mirror_info["min_longitudinal_offset"],
        "max_x": mirror_info["max_longitudinal_offset"],
        "min_y": mirror_info["min_lateral_offset"],
        "max_y": mirror_info["max_lateral_offset"],
        "min_z": mirror_info["min_height_offset"],
        "max_z": mirror_info["max_height_offset"],
    })

    nodes.append(
        ComposableNode(
            package="autoware_pointcloud_preprocessor",
            plugin="autoware::pointcloud_preprocessor::CropBoxFilterComponent",
            name="crop_box_filter_mirror",
            remappings=[
                ("input", "self_cropped/pointcloud"),
                ("output", "mirror_cropped/pointcloud"),
            ],
            parameters=[mirror_cropbox_parameters],
            extra_arguments=[{"use_intra_process_comms": LaunchConfiguration("use_intra_process")}],
        )
    )

    # 3. Above-vehicle crop box. The Ouster returns floating points 1.5-2.4 m high over the hood, up to 0.3 m beyond
    #    the body (sensor rig / reflections). CenterPoint turned them into a pedestrian on the hood, which blocked the
    #    start (surround_obstacle_checker). This box only covers the band above the car, so real obstacles next to
    #    the car keep their lower points.
    above_cropbox_parameters = create_parameter_dict("input_frame", "output_frame")
    above_cropbox_parameters["negative"] = True
    above_cropbox_parameters["processing_time_threshold_sec"] = 0.1
    lateral_margin = float(LaunchConfiguration("above_vehicle_crop_box_lateral_margin").perform(context))
    above_cropbox_parameters.update({
        "min_x": vehicle_info["min_longitudinal_offset"],
        "max_x": vehicle_info["max_longitudinal_offset"],
        "min_y": vehicle_info["min_lateral_offset"] - lateral_margin,
        "max_y": vehicle_info["max_lateral_offset"] + lateral_margin,
        "min_z": float(LaunchConfiguration("above_vehicle_crop_box_min_z").perform(context)),
        "max_z": float(LaunchConfiguration("above_vehicle_crop_box_max_z").perform(context)),
    })

    nodes.append(
        ComposableNode(
            package="autoware_pointcloud_preprocessor",
            plugin="autoware::pointcloud_preprocessor::CropBoxFilterComponent",
            name="crop_box_filter_above_vehicle",
            remappings=[
                ("input", "mirror_cropped/pointcloud"),
                ("output", "above_vehicle_cropped/pointcloud"),
            ],
            parameters=[above_cropbox_parameters],
            extra_arguments=[{"use_intra_process_comms": LaunchConfiguration("use_intra_process")}],
        )
    )

    # 4. Distortion Corrector (Deskewing using Vehicle Twist & IMU)
    nodes.append(
        ComposableNode(
            package="autoware_pointcloud_preprocessor",
            plugin="autoware::pointcloud_preprocessor::DistortionCorrectorComponent",
            name="distortion_corrector_node",
            remappings=[
                ("~/input/twist", "/sensing/vehicle_velocity_converter/twist_with_covariance"),
                ("~/input/imu", "/sensing/imu/imu_data"),
                ("~/input/pointcloud", "above_vehicle_cropped/pointcloud"),
                ("~/output/pointcloud", "rectified/pointcloud"),
            ],
            parameters=[distortion_corrector_node_param],
            extra_arguments=[{"use_intra_process_comms": LaunchConfiguration("use_intra_process")}],
        )
    )

    # 5. Ring Outlier Filter (Outputs directly to /sensing/lidar/concatenated/pointcloud)
    nodes.append(
        ComposableNode(
            package="autoware_pointcloud_preprocessor",
            plugin="autoware::pointcloud_preprocessor::RingOutlierFilterComponent",
            name="ring_outlier_filter",
            remappings=[
                ("input", "rectified/pointcloud"),
                ("output", "/sensing/lidar/concatenated/pointcloud"),
            ],
            parameters=[ring_outlier_filter_node_param, {"output_frame": ""}],
            extra_arguments=[{"use_intra_process_comms": LaunchConfiguration("use_intra_process")}],
        )
    )

    # Load all nodes into the shared pointcloud_container
    load_nodes = LoadComposableNodes(
        composable_node_descriptions=nodes,
        target_container=LaunchConfiguration("pointcloud_container_name"),
    )

    return [load_nodes]


def generate_launch_description():
    # common_sensor_launch was removed upstream (#1938), its param files moved to autoware_sensing_launch
    common_sensor_share_dir = get_package_share_directory("autoware_sensing_launch")

    launch_arguments = [
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("input_frame", default_value="base_link"),
        DeclareLaunchArgument("output_frame", default_value="base_link"),
        DeclareLaunchArgument("use_multithread", default_value="False"),
        DeclareLaunchArgument("use_intra_process", default_value="False"),
        DeclareLaunchArgument("pointcloud_container_name", default_value="pointcloud_container"),
        DeclareLaunchArgument("vehicle_mirror_param_file", description="Path to vehicle mirror param file"),
        DeclareLaunchArgument("above_vehicle_crop_box_min_z", default_value="1.4", description="[m] above base_link"),
        DeclareLaunchArgument("above_vehicle_crop_box_max_z", default_value="2.3", description="[m] above base_link, sensor rig height"),
        DeclareLaunchArgument("above_vehicle_crop_box_lateral_margin", default_value="0.4", description="[m] beyond the body on each side"),
        DeclareLaunchArgument(
            "distortion_correction_node_param_path",
            default_value=os.path.join(common_sensor_share_dir, "config", "distortion_corrector_node.param.yaml"),
        ),
        DeclareLaunchArgument(
            "ring_outlier_filter_node_param_path",
            default_value=os.path.join(common_sensor_share_dir, "config", "ring_outlier_filter_node.param.yaml"),
        ),
    ]

    return launch.LaunchDescription(launch_arguments + [OpaqueFunction(function=launch_setup)])