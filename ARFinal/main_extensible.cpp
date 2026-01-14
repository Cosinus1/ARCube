/**
 * @file main_extensible.cpp
 * @brief Demonstration of extensible AR architecture with MarkerRegistry
 * 
 * This version shows how to use the extensible marker-object system,
 * allowing new markers and 3D objects to be added via configuration
 * without modifying the core codebase.
 * 
 * Compile: cd build && cmake .. && make
 * Run: ./ARFinal --extensible --config ../marker_registry.yaml
 */

#include "Config.hpp"
#include "Exceptions.hpp"
#include "RAII.hpp"
#include "Detection.hpp"
#include "Calibration.hpp"
#include "PoseEstimation.hpp"
#include "Rendering.hpp"
#include "MarkerRegistry.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <chrono>

// Global variables
cv::Mat cameraMatrix, distCoeffs;
int WINDOW_WIDTH = 1280;
int WINDOW_HEIGHT = 720;

void processFrameWithRegistry(const cv::Mat& frame, 
                              MarkerRegistry& registry,
                              RobustPoseEstimator& poseEstimator,
                              const GLProgram& lineProgram,
                              const glm::mat4& projectionGL) {
    
    // Try all enabled markers in registry
    for (auto& pair : registry.getAllPairs()) {
        if (!pair.enabled) continue;
        
        std::vector<cv::Point2f> corners;
        double confidence;
        
        // Detect marker
        if (pair.marker->detect(frame, corners, confidence)) {
            // Get marker dimensions for pose estimation
            cv::Size2f dims = pair.marker->getDimensions();
            
            // Define 3D points (marker plane in mm)
            std::vector<cv::Point3f> objectPoints = {
                cv::Point3f(0, 0, 0),
                cv::Point3f(dims.width, 0, 0),
                cv::Point3f(dims.width, dims.height, 0),
                cv::Point3f(0, dims.height, 0)
            };
            
            // Estimate pose
            cv::Mat rvec, tvec;
            auto metrics = poseEstimator.estimateRobustPose(
                objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec
            );
            
            if (metrics.state != PoseState::POOR) {
                // Convert pose to OpenGL ModelView matrix
                glm::mat4 modelView = getModelViewMatrix(rvec, tvec);
                glm::mat4 mvp = projectionGL * modelView;
                
                // Render associated 3D object
                pair.object->render(mvp, lineProgram);
                
                // Print detection info
                std::cout << "✓ " << pair.name << " | "
                          << pair.marker->getType() << " → " 
                          << pair.object->getName()
                          << " | Confidence: " << confidence
                          << " | Pose: " << metrics.getStateString() << "\n";
            }
        }
    }
}

int main(int argc, char** argv) {
    try {
        std::cout << "=== AR with Extensible Architecture ===\n\n";
        
        // Parse arguments
        std::string registryConfigFile = "marker_registry.yaml";
        std::string cameraConfigFile = "camera.yaml";
        std::string arConfigFile = "ar_config.yaml";
        int cameraId = 0;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--registry" && i + 1 < argc) {
                registryConfigFile = argv[++i];
            } else if (arg == "--camera-config" && i + 1 < argc) {
                cameraConfigFile = argv[++i];
            } else if (arg == "--ar-config" && i + 1 < argc) {
                arConfigFile = argv[++i];
            } else if (arg == "--camera" && i + 1 < argc) {
                cameraId = std::stoi(argv[++i]);
            } else if (arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                          << "Options:\n"
                          << "  --registry FILE      Marker registry config (default: marker_registry.yaml)\n"
                          << "  --camera-config FILE Camera calibration (default: camera.yaml)\n"
                          << "  --ar-config FILE     AR parameters (default: ar_config.yaml)\n"
                          << "  --camera ID          Camera device ID (default: 0)\n"
                          << "  --help               Show this help\n";
                return 0;
            }
        }
        
        // Load configurations
        std::cout << "Loading AR configuration...\n";
        Config::load(arConfigFile);
        Config::printSummary();
        
        std::cout << "\nLoading camera calibration...\n";
        auto calibration = loadCalibration(cameraConfigFile);
        calibration.printSummary();
        cameraMatrix = calibration.camera_matrix;
        distCoeffs = calibration.distortion_coeffs;
        
        // Load marker registry
        std::cout << "\nLoading marker registry...\n";
        MarkerRegistry registry;
        if (!registry.loadFromConfig(registryConfigFile)) {
            std::cerr << "⚠ Warning: Could not load registry, using default A4+Cube\n";
            
            // Fallback: register default A4 + Cube
            auto defaultMarker = std::make_shared<A4MarkerDetector>();
            auto defaultObject = std::make_shared<CubeObject>(50.0f, glm::vec3(0, 1, 0));
            registry.registerPair("A4_Cube_Fallback", defaultMarker, defaultObject);
        }
        registry.printRegistry();
        
        // Open camera
        VideoCaptureRAII cap(cameraId);
        cv::Mat frame;
        cap >> frame;
        require(!frame.empty(), "Cannot read from camera");
        
        int frameWidth = frame.cols;
        int frameHeight = frame.rows;
        std::cout << "\nCamera resolution: " << frameWidth << "x" << frameHeight << "\n";
        
        // Scale calibration if needed
        if (calibration.image_width != frameWidth || calibration.image_height != frameHeight) {
            calibration.scaleToSize(cv::Size(frameWidth, frameHeight));
            cameraMatrix = calibration.camera_matrix;
        }
        
        // Initialize GLFW and create window
        GLFWManager glfwMgr(WINDOW_WIDTH, WINDOW_HEIGHT, "AR - Extensible Architecture");
        GLFWwindow* window = glfwMgr.getWindow();
        
        // Initialize GLEW
        glewExperimental = GL_TRUE;
        GLenum glewStatus = glewInit();
        require(glewStatus == GLEW_OK, 
                std::string("GLEW init failed: ") + (const char*)glewGetErrorString(glewStatus));
        
        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << "\n";
        std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n\n";
        
        // Setup OpenGL
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glLineWidth(2.0f);
        
        // Create shader programs
        GLProgram bgProgram = linkProgram({
            compileShader(BG_VS, GL_VERTEX_SHADER),
            compileShader(BG_FS, GL_FRAGMENT_SHADER)
        });
        
        GLProgram lineProgram = linkProgram({
            compileShader(LINE_VS, GL_VERTEX_SHADER),
            compileShader(LINE_GS, GL_GEOMETRY_SHADER),
            compileShader(LINE_FS, GL_FRAGMENT_SHADER)
        });
        
        // Create background quad
        MeshRAII bgQuad = createBackgroundQuadRAII();
        GLTexture bgTexture;
        
        // Projection matrix
        glm::mat4 projectionGL = projectionFromCV(cameraMatrix, frameWidth, frameHeight, 0.1f, 1000.0f);
        
        // Pose estimator
        RobustPoseEstimator poseEstimator;
        
        // FPS counter
        auto lastTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        
        std::cout << "Starting main loop...\n";
        std::cout << "Press ESC to quit\n\n";
        
        // Main loop
        while (!glfwWindowShouldClose(window)) {
            // Capture frame
            cap >> frame;
            if (frame.empty()) break;
            
            // Update background texture
            cv::Mat flipped;
            cv::flip(frame, flipped, 0);
            bgTexture.uploadRGB(flipped.data, frameWidth, frameHeight);
            
            // Clear buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            // Render background
            bgProgram.use();
            bgTexture.bind(0);
            bgProgram.setUniform1i("uTexture", 0);
            bgQuad.bind();
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            bgQuad.unbind();
            
            // Process all markers in registry
            processFrameWithRegistry(frame, registry, poseEstimator, lineProgram, projectionGL);
            
            // Swap buffers
            glfwSwapBuffers(window);
            glfwPollEvents();
            
            // FPS calculation
            frameCount++;
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastTime).count();
            if (elapsed >= 1.0) {
                std::cout << "FPS: " << frameCount << "\n";
                frameCount = 0;
                lastTime = currentTime;
            }
            
            // ESC to quit
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                break;
            }
        }
        
        std::cout << "\n=== Shutdown complete ===\n";
        return 0;
        
    } catch (const ARException& e) {
        std::cerr << "AR Error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
