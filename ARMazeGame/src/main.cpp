/**
 * @file main.cpp
 * @brief AR Maze Game - Main Application
 * 
 * This application detects a flat surface (like A4 paper), renders a 3D maze on it,
 * and simulates ball physics based on surface tilt. Guide the ball to the goal!
 * 
 * Usage: ./ARMazeGame
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <array>
#include <algorithm>

#include "Utils.hpp"
#include "Detection.hpp"
#include "Drawing.hpp"
#include "Quat.hpp"
#include "Pose.hpp"
#include "Calibration.hpp"
#include "Physics.hpp"
#include "Maze.hpp"
#include "Renderer.hpp"
#include "GUI.hpp"

// Global configuration
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const std::string WINDOW_NAME = "AR Maze Game";

// Physical size of the detected plane (A4 paper in mm)
const float PLANE_WIDTH = 210.0f;   // A4 width
const float PLANE_HEIGHT = 297.0f;  // A4 height

// Game configuration
const float BALL_RADIUS = 8.0f;
const float WALL_HEIGHT = 25.0f;
const float WALL_THICKNESS = 5.0f;

class ARMazeGame {
public:
    ARMazeGame() 
        : running(true)
        , frameCount(0)
        , lastTime(std::chrono::high_resolution_clock::now())
        , deltaTime(0.016f)
        , hasPrevPose(false)
        , poseAlpha(0.45)
        , quadAlpha(0.6f)
        , hadSmoothQuad(false)
    {
        gui = std::make_unique<GUIManager>(WINDOW_WIDTH, WINDOW_HEIGHT);
        
        // Setup callbacks
        gui->setOnStartGame([this]() { startGame(); });
        gui->setOnResetGame([this]() { resetGame(); });
        gui->setOnQuit([this]() { running = false; });
    }
    
    bool initialize(const std::filesystem::path& dataRoot) {
        this->dataRoot = dataRoot;
        
        // Find available sources
        imageDir = dataRoot / "image";
        videoDir = dataRoot / "video";
        yamlDir = dataRoot / "yaml";
        
        // List available files
        auto images = listFilesInDirectory(imageDir, {".jpg", ".jpeg", ".png", ".bmp"});
        auto videos = listFilesInDirectory(videoDir, {".mp4", ".avi", ".mov", ".mkv"});
        auto yamls = listFilesInDirectory(yamlDir, {".yaml", ".yml"});
        auto webcams = getAvailableWebcams(5);
        
        gui->setImageFiles(images);
        gui->setVideoFiles(videos);
        gui->setYamlFiles(yamls);
        gui->setWebcams(webcams);
        
        std::cout << "Found " << images.size() << " images, " 
                  << videos.size() << " videos, "
                  << webcams.size() << " webcams\n";
        
        // Create window
        cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(WINDOW_NAME, mouseCallback, gui.get());
        
        return true;
    }
    
    void run() {
        cv::Mat frame;
        
        while (running) {
            updateDeltaTime();
            
            if (gui->getCurrentMode() == AppMode::PLAYING && capture.isOpened()) {
                // Game mode - process video/webcam
                processGameFrame(frame);
            } else {
                // Menu mode
                gui->render(frame);
            }
            
            cv::imshow(WINDOW_NAME, frame);
            
            int key = cv::waitKey(1);
            handleKey(key);
            
            frameCount++;
        }
        
        cleanup();
    }

private:
    void startGame() {
        std::cout << "Starting game...\n";
        
        // Open source
        bool sourceOpened = false;
        
        if (gui->isWebcamMode()) {
            int camIdx = gui->getSelectedWebcam();
            capture.open(camIdx);
            sourceOpened = capture.isOpened();
            isImage = false;
            std::cout << "Opening webcam " << camIdx << ": " << (sourceOpened ? "OK" : "FAILED") << "\n";
        } else if (gui->isVideoMode()) {
            std::filesystem::path videoPath = videoDir / gui->getSelectedSource();
            capture.open(videoPath.string());
            sourceOpened = capture.isOpened();
            isImage = false;
            std::cout << "Opening video " << videoPath << ": " << (sourceOpened ? "OK" : "FAILED") << "\n";
        } else if (gui->isImageMode()) {
            std::filesystem::path imagePath = imageDir / gui->getSelectedSource();
            staticImage = cv::imread(imagePath.string());
            sourceOpened = !staticImage.empty();
            isImage = true;
            std::cout << "Opening image " << imagePath << ": " << (sourceOpened ? "OK" : "FAILED") << "\n";
        }
        
        if (!sourceOpened) {
            std::cerr << "Failed to open source!\n";
            gui->resetSelection();
            return;
        }
        
        // Get frame size
        cv::Mat testFrame;
        if (isImage) {
            testFrame = staticImage;
        } else {
            capture >> testFrame;
            if (testFrame.empty()) {
                std::cerr << "Failed to read test frame!\n";
                gui->resetSelection();
                return;
            }
        }
        
        frameWidth = testFrame.cols;
        frameHeight = testFrame.rows;
        std::cout << "Frame size: " << frameWidth << "x" << frameHeight << "\n";
        
        // Load or create calibration
        if (!loadOrCreateCalibration()) {
            std::cerr << "Calibration failed!\n";
            gui->resetSelection();
            return;
        }
        
        // Initialize physics and maze
        initializeGame();
        
        std::cout << "Game started! Maze: " << MazeGenerator::getMazeTypeName(gui->getSelectedMaze()) << "\n";
    }
    
    bool loadOrCreateCalibration() {
        // Try to load existing calibration
        std::vector<std::string> yamlFiles = listFilesInDirectory(yamlDir, {".yaml", ".yml"});
        
        for (const auto& yamlFile : yamlFiles) {
            std::filesystem::path yamlPath = yamlDir / yamlFile;
            cv::Mat K, D;
            cv::Size calibSize;
            
            if (loadCalibration(yamlPath.string(), K, D, calibSize)) {
                // Check if calibration size matches or can be scaled
                if (calibSize.width > 0 && calibSize.height > 0) {
                    cameraMatrix = scaleIntrinsics(K, calibSize, cv::Size(frameWidth, frameHeight));
                } else {
                    cameraMatrix = K;
                }
                distCoeffs = D;
                std::cout << "Loaded calibration from: " << yamlPath << "\n";
                return true;
            }
        }
        
        // Create default calibration
        std::cout << "No calibration found, using default\n";
        createDefaultCalibration(frameWidth, frameHeight, cameraMatrix, distCoeffs);
        return true;
    }
    
    void initializeGame() {
        // Calculate play area (scaled from plane dimensions)
        float halfW = PLANE_WIDTH * 0.45f;  // Leave margin
        float halfH = PLANE_HEIGHT * 0.45f;
        
        // Initialize physics
        physics.initialize(halfW, halfH, BALL_RADIUS);
        physics.clearWalls();
        
        // Generate maze
        MazeGenerator generator;
        MazeType mazeType = gui->getSelectedMaze();
        auto walls = generator.generateMaze(mazeType, halfW, halfH, WALL_THICKNESS);
        
        for (const auto& wall : walls) {
            physics.addWall(wall);
        }
        
        // Set goal
        Goal goal = generator.getGoalPosition(mazeType, halfW, halfH);
        physics.setGoal(goal);
        
        // Reset ball position
        physics.reset();
        
        // Initialize game state
        gameState.startGame(mazeType);
        
        // Reset pose tracking
        hasPrevPose = false;
        hadSmoothQuad = false;
    }
    
    void resetGame() {
        physics.reset();
        gameState.resetGame();
        hasPrevPose = false;
        hadSmoothQuad = false;
    }
    
    void processGameFrame(cv::Mat& displayFrame) {
        cv::Mat frame;
        
        if (isImage) {
            frame = staticImage.clone();
        } else {
            capture >> frame;
            if (frame.empty()) {
                // Video ended - loop or stop
                if (gui->isVideoMode()) {
                    capture.set(cv::CAP_PROP_POS_FRAMES, 0);
                    capture >> frame;
                }
                if (frame.empty()) {
                    gui->resetSelection();
                    return;
                }
            }
        }
        
        // Convert to grayscale for detection
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // Detect surface
        std::vector<cv::Point2f> quad;
        double confidence = 0.0;
        bool detected = detectSheetAdaptive(frame, quad, confidence);
        
        // Fallback to simple detection
        if (!detected) {
            detected = findLargestQuad(gray, quad);
            if (detected) confidence = 0.6;
        }
        
        displayFrame = frame.clone();
        
        if (detected && quad.size() == 4) {
            // Smooth corners
            std::array<cv::Point2f, 4> quadSm;
            if (!hadSmoothQuad) {
                for (int i = 0; i < 4; i++) {
                    smoothQuad[i] = quad[i];
                    quadSm[i] = quad[i];
                }
                hadSmoothQuad = true;
            } else {
                for (int i = 0; i < 4; i++) {
                    smoothQuad[i] = quadAlpha * smoothQuad[i] + (1.0f - quadAlpha) * quad[i];
                    quadSm[i] = smoothQuad[i];
                }
            }
            
            // Draw detected quad
            std::vector<cv::Point2f> quadVec(quadSm.begin(), quadSm.end());
            drawLabeledQuad(displayFrame, quadVec);
            
            // Estimate pose
            cv::Mat rvec, tvec;
            bool poseOK = poseFromQuadHomography(quadVec, cameraMatrix, distCoeffs, 
                                                  PLANE_HEIGHT, rvec, tvec);
            
            if (poseOK) {
                // Smooth pose
                cv::Mat rDisp, tDisp;
                if (hasPrevPose) {
                    Quat qPrev = rotvecToQuat(rvecPrev);
                    Quat qCurr = rotvecToQuat(rvec);
                    Quat qMix = quatSlerp(qPrev, qCurr, poseAlpha);
                    rDisp = quatToRotvec(qMix);
                    tDisp = tvecPrev * (1.0 - poseAlpha) + tvec * poseAlpha;
                } else {
                    rDisp = rvec.clone();
                    tDisp = tvec.clone();
                    hasPrevPose = true;
                }
                
                rvecPrev = rDisp.clone();
                tvecPrev = tDisp.clone();
                
                // Update physics
                if (gameState.getState() == GameState::State::PLAYING) {
                    physics.update(rDisp, deltaTime);
                    gameState.update(deltaTime);
                    
                    // Check win condition
                    if (physics.isAtGoal()) {
                        gameState.win();
                        std::cout << "YOU WIN! Time: " << gameState.getElapsedTime() << "s\n";
                    }
                }
                
                // Render game elements
                renderer.render(displayFrame, cameraMatrix, distCoeffs, 
                               rDisp, tDisp, physics, gameState);
                
                // Draw axes for reference
                drawAxes(displayFrame, cameraMatrix, distCoeffs, rDisp, tDisp, 50.0f);
            }
        } else {
            // No detection - show searching message
            renderer.renderHUD(displayFrame, gameState, physics.getTilt(), false);
            
            cv::putText(displayFrame, "Looking for surface...", 
                       cv::Point(displayFrame.cols / 2 - 100, displayFrame.rows / 2),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            cv::putText(displayFrame, "Place a flat rectangular surface in view",
                       cv::Point(displayFrame.cols / 2 - 180, displayFrame.rows / 2 + 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        }
    }
    
    void handleKey(int key) {
        if (key < 0) return;
        
        if (gui->handleKey(key)) return;
        
        if (gui->getCurrentMode() == AppMode::PLAYING) {
            if (key == 'r' || key == 'R') {
                resetGame();
            } else if (key == 'm' || key == 'M') {
                // Return to menu
                cleanup();
                gui->resetSelection();
            }
        }
    }
    
    void updateDeltaTime() {
        auto now = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        
        // Clamp delta time
        deltaTime = std::min(deltaTime, 0.1f);
    }
    
    void cleanup() {
        if (capture.isOpened()) {
            capture.release();
        }
        staticImage.release();
    }
    
    // Member variables
    bool running;
    int frameCount;
    std::chrono::high_resolution_clock::time_point lastTime;
    float deltaTime;
    
    // Paths
    std::filesystem::path dataRoot;
    std::filesystem::path imageDir;
    std::filesystem::path videoDir;
    std::filesystem::path yamlDir;
    
    // GUI
    std::unique_ptr<GUIManager> gui;
    
    // Video/image source
    cv::VideoCapture capture;
    cv::Mat staticImage;
    bool isImage = false;
    int frameWidth = 640;
    int frameHeight = 480;
    
    // Camera calibration
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    
    // Game systems
    PhysicsSimulator physics;
    GameState gameState;
    MazeRenderer renderer;
    
    // Pose tracking
    bool hasPrevPose;
    cv::Mat rvecPrev, tvecPrev;
    double poseAlpha;
    
    // Corner smoothing
    float quadAlpha;
    std::array<cv::Point2f, 4> smoothQuad;
    bool hadSmoothQuad;
};

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "       AR MAZE GAME\n";
    std::cout << "========================================\n\n";
    
    // Find data directory
    std::filesystem::path exePath = std::filesystem::path(argv[0]);
    std::filesystem::path dataRoot = resolveDataRoot(exePath);
    
    std::cout << "Data directory: " << dataRoot << "\n\n";
    
    // Create and run game
    ARMazeGame game;
    
    if (!game.initialize(dataRoot)) {
        std::cerr << "Failed to initialize game!\n";
        return -1;
    }
    
    game.run();
    
    std::cout << "\nGame ended. Goodbye!\n";
    return 0;
}
