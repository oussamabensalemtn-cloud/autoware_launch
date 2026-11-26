cmake_minimum_required(VERSION 3.5)
project(ioniq5_vehicle_launch)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ControllerTrajectory.msg"
  DEPENDENCIES std_msgs
)

# Install the Python nodes into the package's binary library path
install(PROGRAMS
  launch/vehicle_status_publisher.py
  launch/autobox_trajectory_bridge.py
  launch/autobox_udp_sender.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
)