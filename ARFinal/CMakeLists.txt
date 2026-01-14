cmake_minimum_required(VERSION 3.10)
project(ARFinal)

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Source files
add_executable(ARFinal 
    main.cpp
    Config.cpp
    RAII.cpp
    Detection.cpp
    Calibration.cpp
    PoseEstimation.cpp
    Rendering.cpp
    MarkerRegistry.cpp
)

target_compile_features(ARFinal PRIVATE cxx_std_17)
target_link_libraries(ARFinal ${OpenCV_LIBS} ${OPENGL_LIBRARIES} GLEW::GLEW glfw)