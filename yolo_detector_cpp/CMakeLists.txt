cmake_minimum_required(VERSION 3.8)
project(yolo_detector_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR
  NvInfer.h
  PATHS /usr/include/aarch64-linux-gnu
  REQUIRED
)

find_library(TENSORRT_LIBRARY
  nvinfer
  PATHS /usr/lib/aarch64-linux-gnu
  REQUIRED
)

add_executable(
  yolo_detector_node
  src/yolo_detector_node.cpp
  src/tensorrt_engine.cpp
)

ament_target_dependencies(
  yolo_detector_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)

target_include_directories(
  yolo_detector_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(
  yolo_detector_node
  ${OpenCV_LIBRARIES}
  ${TENSORRT_LIBRARY}
  CUDA::cudart
)

install(
  TARGETS yolo_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()