cmake_minimum_required(VERSION 3.5)
project(ioniq5_sensor_kit_launch)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(PROGRAMS
launch/twist_to_velocity_report.py
launch/ouster_format_converter.py
launch/tf_to_gnssinsorient_converter.py
launch/camera_info_converter.py
DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE
  launch
  data
  config
)
