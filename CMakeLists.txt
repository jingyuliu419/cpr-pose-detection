cmake_minimum_required(VERSION 3.14)
project(multi_camera_ros_sync)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-Wno-deprecated-declarations)

# --- OpenCV ---
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# OpenCV CUDA modules
set(OpenCV_LIB_COMPONENTS
    ${OpenCV_LIBS}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_cudaimgproc
    opencv_cudawarping
)

# --- CUDA ---
find_package(CUDA REQUIRED)

# --- TensorRT ---
set(TENSORRT_ROOT "/usr/include/tensorrt")
include_directories(${TENSORRT_ROOT})
link_directories(/usr/lib/x86_64-linux-gnu)
set(TENSORRT_LIBS
    nvinfer
    nvonnxparser
    cudart
)

# --- FFmpeg + ROS2 ---
find_package(rclcpp REQUIRED)
find_package(X11 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
    libswscale
)

# --- Include Directories ---
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  /usr/local/cuda/include
  ${OpenCV_INCLUDE_DIRS}
)

# --- CUDA Kernels ---
cuda_add_library(nv12_cuda STATIC src/nv12_cuda.cu)

# --- Executable Sources ---
add_executable(multi_cam_node
  src/main.cpp
  src/video_stream.cpp
  src/ros2_time_sync.cpp
  src/camera_calibrator.cpp
  src/yolov5_trt_detector.cpp
  src/trt_logger.cpp
  src/config_loader.cpp
  src/ui_display_thread.cpp
  src/rtmpose_trt.cpp
  src/charuco_camera_manager.cpp
  src/multi_cam_calibrator.cpp
  src/Triangulator.cpp  
)

# --- Link Libraries ---
target_link_libraries(multi_cam_node
  nv12_cuda
  rclcpp::rclcpp
  ${OpenCV_LIB_COMPONENTS}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBS}
  PkgConfig::FFMPEG
  ${X11_LIBRARIES}
)
