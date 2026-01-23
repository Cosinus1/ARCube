# AR Maze Game

An augmented reality maze game where you guide a ball to the goal by tilting a physical surface!

## Features

- **Surface Detection**: Automatically detects flat rectangular surfaces (like A4 paper)
- **3D Maze Rendering**: Renders a 3D maze on the detected surface
- **Physics Simulation**: Realistic ball physics with gravity and collisions
- **Multiple Maze Types**: 5 different maze difficulties (Simple, Medium, Hard, Spiral, Zigzag)
- **Multiple Input Sources**: Support for webcam, video files, and images

## Requirements

- OpenCV 4.x
- CMake 3.10+
- C++17 compatible compiler

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./ARMazeGame
```

## How to Play

1. **Select Input Mode**: Choose between Image, Video, or Webcam
2. **Select Source**: Pick your specific image/video file or webcam
3. **Select Maze**: Choose your difficulty level
4. **Play**: 
   - Point your camera at a flat rectangular surface (A4 paper works great!)
   - Tilt the surface to roll the ball
   - Guide the ball to the green GOAL area
   - Try to complete the maze as fast as possible!

## Controls

- **R**: Reset the ball to starting position
- **M**: Return to main menu
- **Q/ESC**: Quit the game

## Tips

- Use a well-lit area for better detection
- A contrasting background helps (e.g., white paper on dark table)
- Keep the camera stable for smoother gameplay
- Start with "Simple" maze to learn the controls

## File Structure

```
ARMazeGame/
├── CMakeLists.txt
├── README.md
├── include/          # Header files
│   ├── Utils.hpp
│   ├── Detection.hpp
│   ├── Drawing.hpp
│   ├── Quat.hpp
│   ├── Pose.hpp
│   ├── Calibration.hpp
│   ├── Physics.hpp
│   ├── Maze.hpp
│   ├── Renderer.hpp
│   └── GUI.hpp
├── src/              # Source files
│   ├── main.cpp
│   ├── Utils.cpp
│   ├── Detection.cpp
│   ├── Drawing.cpp
│   ├── Quat.cpp
│   ├── Pose.cpp
│   ├── Calibration.cpp
│   ├── Physics.cpp
│   ├── Maze.cpp
│   ├── Renderer.cpp
│   └── GUI.cpp
└── data/             # Data files
    ├── yaml/         # Camera calibration files
    ├── image/        # Test images (add your own)
    └── video/        # Test videos (add your own)
```

## Camera Calibration

For best results, use a calibrated camera. You can:
1. Use the default calibration (works reasonably well)
2. Add your own calibration YAML file to `data/yaml/`

## Adding Custom Content

- **Images**: Add `.jpg`, `.jpeg`, `.png`, or `.bmp` files to `data/image/`
- **Videos**: Add `.mp4`, `.avi`, `.mov`, or `.mkv` files to `data/video/`
- **Calibration**: Add `.yaml` calibration files to `data/yaml/`

## Troubleshooting

- **Surface not detected**: Ensure good lighting and clear edges
- **Jittery movement**: Try smoother surface movements
- **Wrong orientation**: Make sure the camera can see all 4 corners

Enjoy the game!
