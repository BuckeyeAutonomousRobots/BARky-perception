cmake_minimum_required(VERSION 3.8)
project(stereo_camera_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(
  stereo_camera_node
  src/stereo_camera_node.cpp
)

ament_target_dependencies(
  stereo_camera_node
  rclcpp
  sensor_msgs
  cv_bridge
  std_msgs
)

target_include_directories(
  stereo_camera_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  stereo_camera_node
  ${OpenCV_LIBRARIES}
)

install(
  TARGETS stereo_camera_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()