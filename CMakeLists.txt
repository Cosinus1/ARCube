cmake_minimum_required(VERSION 3.10)
project(AROne LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SRC_FILES
    src/Utils.cpp
    src/Detection.cpp
    src/Drawing.cpp
    src/Quat.cpp
    src/Pose.cpp
    src/Calibration.cpp
    src/main.cpp
)

add_executable(AROne ${SRC_FILES})
target_link_libraries(AROne ${OpenCV_LIBS})
