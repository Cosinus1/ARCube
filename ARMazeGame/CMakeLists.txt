cmake_minimum_required(VERSION 3.10)
project(ARMazeGame LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SRC_FILES
    src/main.cpp
    src/Utils.cpp
    src/Detection.cpp
    src/Drawing.cpp
    src/Quat.cpp
    src/Pose.cpp
    src/Calibration.cpp
    src/Physics.cpp
    src/Maze.cpp
    src/Renderer.cpp
    src/GUI.cpp
)

# Create executable
add_executable(ARMazeGame ${SRC_FILES})

# Link libraries
target_link_libraries(ARMazeGame 
    ${OpenCV_LIBS} 
    ${OPENGL_LIBRARIES} 
    GLEW::GLEW 
    glfw
)

# Copy data files to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})
