cmake_minimum_required(VERSION 2.8)

project(RSTrack)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_BUILD_TYPE "Debug")

#寻找OpenCV库
find_package(OpenCV REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 设置显示模块运行需要的python环境路径
set(PYTHON_INCLUDE_DIR "/home/simulator/anaconda3/envs/HRS/include/python3.8")
set(NUMPY_INCLUDE_DIR "/home/simulator/anaconda3/envs/HRS/lib/python3.8/site-packages/numpy/core/include")
set(PYTHON_LIB "/home/simulator/anaconda3/envs/HRS/lib/libpython3.8.so")

#添加程序库
add_library(${PROJECT_NAME} SHARED
src/VideoCapture.cpp
src/Group.cpp
src/HomographyMap.cpp
src/HRSmatcher.cpp
src/FrameDrawer.cpp
#src/Viewer.cpp
)

#链接OpenCV库
target_link_libraries(${PROJECT_NAME}
                        /usr/lib/x86_64-linux-gnu/libtiff.so.5
                        ${OpenCV_LIBS}
                        ${PYTHON_LIB})

#添加头文件
include_directories("/usr/include/eigen3")   # Eigen
include_directories(${OpenCV_INCLUDE_DIRS})        # Opencv
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})

# 生成执行文件
add_executable(main main.cpp)
target_link_libraries(main ${PROJECT_NAME})




