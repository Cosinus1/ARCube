
cmake_minimum_required(VERSION 3.10)
project(ARDetect)

find_package(OpenCV REQUIRED)

add_executable(ARDetect main.cpp)
target_compile_features(ARDetect PRIVATE cxx_std_17)
target_link_libraries(ARDetect ${OpenCV_LIBS})
