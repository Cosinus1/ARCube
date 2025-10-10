
cmake_minimum_required(VERSION 3.10)
project(ARCube)

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)


add_executable(ARCube main.cpp)
target_compile_features(ARCube PRIVATE cxx_std_17)
target_link_libraries(ARCube ${OpenCV_LIBS} ${OPENGL_LIBRARIES} GLEW::GLEW glfw)
