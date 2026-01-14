/**
 * @file main.cpp (Refactored)
 * @brief Advanced AR Cube Detection and Rendering on A4 Sheet - Main Application
 * @author Olivier Deruelle
 * @date 2025
 * 
 * @details This is the refactored main application file that coordinates all AR modules.
 * The implementation has been reorganized into separate modules for better maintainability:
 * - Config: Centralized configuration management
 * - Exceptions: Custom exception types
 * - RAII: Resource management wrappers
 * - Detection: A4 sheet detection algorithms  
 * - Calibration: Camera calibration utilities
 * - PoseEstimation: Robust pose estimation
 * - Rendering: OpenGL rendering utilities
 * 
 * @section usage Usage Examples
 * @code
 * ./ARFinal 0 camera.yaml                      // webcam
 * ./ARFinal image.jpg camera.yaml              // single image
 * ./ARFinal video.mp4 camera.yaml              // video file
 * ./ARFinal mire.jpg                           // automatic calibration
 * @endcode
 */

// clear ; cd ARFinal/build/ ; cmake .. -Wno-dev ; cmake --build . ; ./ARFinal "../../data/Video_AR_1.mp4" ; cd ../..
// clear ; cd ARFinal/build/ ; ./ARFinal "../../data/Video_AR_1.mp4" ; cd ../..

#include <iostream>
#include <vector>
#include <array>
#include <iomanip>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Our modules
#include "Config.hpp"
#include "Exceptions.hpp"
#include "RAII.hpp"
#include "Detection.hpp"
#include "Calibration.hpp"
#include "PoseEstimation.hpp"
#include "Rendering.hpp"

/**
 * @brief Main AR application entry point
 */
int main(int argc, char** argv) {
    try {
        // ============= Argument Parsing =============
        if (argc < 2) {
            std::cout << "Usage: ./ARFinal [0 | image.jpg | video.mp4] [camera_yaml] [config_yaml]\n";
            return -1;
        }
        
        std::string source = argv[1];
        std::string cameraYaml = (argc >= 3) ? argv[2] : std::string("");
        std::string configYaml = (argc >= 4) ? argv[3] : std::string("../../data/ar_config.yaml");
        
        // Load configuration
        Config::load(configYaml);
        if (Config::runtime.verbose) Config::printSummary();
        
        // ============= Automatic Calibration Mode =============
        if (source.find("mire") != std::string::npos) {
            cv::Size boardSize(Config::calibration.chessboardCols, Config::calibration.chessboardRows);
            float squareSize = Config::calibration.chessboardSquareSize;
            
            std::vector<cv::Mat> calibration_images;
            
            std::string ext = source.substr(source.find_last_of('.'));
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
                cv::VideoCapture cap(source);
                if (!cap.isOpened()) {
                    std::cerr << "Cannot open calibration video: " << source << "\n";
                    return -1;
                }
                
                cv::Mat frame;
                int frameCount = 0;
                int maxFrames = 30;
                
                std::cout << "Extracting calibration frames from video...\n";
                while (cap.read(frame) && frameCount < maxFrames) {
                    if (frame.empty()) continue;
                    if (frameCount % 3 == 0) {
                        calibration_images.push_back(frame.clone());
                    }
                    frameCount++;
                }
                std::cout << "Extracted " << calibration_images.size() << " frames for calibration.\n";
            } else {
                cv::Mat frame = cv::imread(source, cv::IMREAD_COLOR);
                if (frame.empty()) {
                    std::cerr << "Cannot open calibration image: " << source << "\n";
                    return -1;
                }
                calibration_images.push_back(frame);
            }
            
            if (calibration_images.empty()) {
                std::cerr << "No calibration images found.\n";
                return -1;
            }
            
            CameraCalibration auto_calib = performAutoCalibration(calibration_images, boardSize, squareSize);
            
            if (!auto_calib.is_valid) {
                std::cerr << "Automatic calibration failed.\n";
                return -1;
            }
            
            saveCalibration("../../data/config.yaml", auto_calib);
            auto_calib.printSummary();
            
            return 0;
        }
        
        // ============= Load Initial Frame =============
        cv::Mat frame;
        bool isImage = false;
        
        frame = cv::imread(source);
        if (!frame.empty()) {
            isImage = true;
        } else if (source != "0") {
            cv::VideoCapture testCap(source);
            if (!testCap.isOpened()) {
                std::cerr << "Impossible d'ouvrir " << source << "\n";
                return -1;
            }
            testCap >> frame;
            if (frame.empty()) {
                std::cerr << "Impossible de récupérer une frame initiale\n";
                return -1;
            }
            testCap.release();
        } else {
            cv::VideoCapture testCap(0);
            if (!testCap.isOpened()) {
                std::cerr << "Webcam non accessible\n";
                return -1;
            }
            testCap >> frame;
            if (frame.empty()) {
                std::cerr << "Impossible de récupérer une frame initiale\n";
                return -1;
            }
            testCap.release();
        }
        
        int width = frame.cols, height = frame.rows;
        int maxWinW = Config::render.maxWindowW, maxWinH = Config::render.maxWindowH;
        int winW = std::min(width, maxWinW);
        int winH = std::min(height, maxWinH);
        
        // ============= Load Calibration =============
        loadCalibration(cameraYaml, width, height);
        
        // ============= Define A4 Sheet Parameters =============
        const float A4_w = 210.0f;
        const float A4_h = 297.0f;
        std::vector<cv::Point3f> objectPoints = {
            {-A4_w/2.0f,  A4_h/2.0f, 0.0f},  // TL
            { A4_w/2.0f,  A4_h/2.0f, 0.0f},  // TR
            { A4_w/2.0f, -A4_h/2.0f, 0.0f},  // BR
            {-A4_w/2.0f, -A4_h/2.0f, 0.0f}   // BL
        };
        
        float cubeHalf = std::min(A4_w, A4_h) * 0.12f;
        float cubeH = std::min(A4_w, A4_h) * 0.12f;
        
        // ============= Prepare Undistort Maps =============
        cv::Mat undistortMap1, undistortMap2;
        {
            cv::Size sz(width, height);
            cv::Mat newK = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, sz, 0.0, sz);
            cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newK, sz, CV_16SC2, 
                                       undistortMap1, undistortMap2);
            cameraMatrix = newK.clone();
            distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
        }
        
        // ============= Initialize OpenGL =============
        GLFWManager glfw_manager(winW, winH, "AR - Cube on A4");
        GLFWwindow* window = glfw_manager.getWindow();
        
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            throw std::runtime_error("GLEW init failed");
        }
        glGetError();
        
        // ============= Create Shaders =============
        GLShader bgVS(GL_VERTEX_SHADER, BG_VS);
        GLShader bgFS(GL_FRAGMENT_SHADER, BG_FS);
        if (!bgVS.isValid() || !bgFS.isValid()) {
            throw std::runtime_error("Background shader compilation failed");
        }
        GLProgram bgProgram({bgVS.getId(), bgFS.getId()});
        if (!bgProgram.isValid()) {
            throw std::runtime_error("Background program linking failed");
        }
        
        GLShader lineVS(GL_VERTEX_SHADER, LINE_VS);
        GLShader lineGS(GL_GEOMETRY_SHADER, LINE_GS);
        GLShader lineFS(GL_FRAGMENT_SHADER, LINE_FS);
        if (!lineVS.isValid() || !lineGS.isValid() || !lineFS.isValid()) {
            throw std::runtime_error("Line shader compilation failed");
        }
        GLProgram lineProgram({lineVS.getId(), lineGS.getId(), lineFS.getId()});
        if (!lineProgram.isValid()) {
            throw std::runtime_error("Line program linking failed");
        }
        
        GLShader markerVS(GL_VERTEX_SHADER, MARKER_VS);
        GLShader markerFS(GL_FRAGMENT_SHADER, MARKER_FS);
        GLProgram markerProgram({markerVS.getId(), markerFS.getId()});
        if (!markerProgram.isValid()) {
            throw std::runtime_error("Marker program linking failed");
        }
        
        GLShader planeVS(GL_VERTEX_SHADER, PLANE_VS);
        GLShader planeFS(GL_FRAGMENT_SHADER, PLANE_FS);
        GLProgram planeProgram({planeVS.getId(), planeFS.getId()});
        if (!planeProgram.isValid()) {
            throw std::runtime_error("Plane program linking failed");
        }

        GLShader planeLightVS(GL_VERTEX_SHADER, PLANE_LIGHT_VS);
        GLShader planeLightFS(GL_FRAGMENT_SHADER, PLANE_LIGHT_FS);
        GLProgram planeLightProgram({planeLightVS.getId(), planeLightFS.getId()});
        if (!planeLightProgram.isValid()) {
            throw std::runtime_error("Plane light program linking failed");
        }

        GLShader wallLightVS(GL_VERTEX_SHADER, WALL_LIGHT_VS);
        GLShader wallLightFS(GL_FRAGMENT_SHADER, WALL_LIGHT_FS);
        GLProgram wallLightProgram({wallLightVS.getId(), wallLightFS.getId()});
        if (!wallLightProgram.isValid()) {
            throw std::runtime_error("Wall light program linking failed");
        }

        GLShader sphereVS(GL_VERTEX_SHADER, SPHERE_VS);
        GLShader sphereFS(GL_FRAGMENT_SHADER, SPHERE_FS);
        GLProgram sphereProgram({sphereVS.getId(), sphereFS.getId()});
        if (!sphereProgram.isValid()) {
            throw std::runtime_error("Sphere program linking failed");
        }
        
        // ============= Create Plane Meshes =============
        MeshRAII planeWire, planeFill;
        {
            float pw = A4_w/2.f, ph = A4_h/2.f;
            float planeVerts[] = {
                -pw, -ph, 0.f,
                 pw, -ph, 0.f,
                 pw,  ph, 0.f,
                -pw,  ph, 0.f
            };
            GLuint outlineIdx[] = {0,1, 1,2, 2,3, 3,0};
            GLuint fillIdx[] = {0,1,2, 0,2,3};
            planeWire.initWithIndices(planeVerts, sizeof(planeVerts),
                                      outlineIdx, sizeof(outlineIdx),
                                      (GLsizei)(sizeof(outlineIdx)/sizeof(outlineIdx[0])), [](){
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
            });
            planeFill.initWithIndices(planeVerts, sizeof(planeVerts),
                                      fillIdx, sizeof(fillIdx),
                                      (GLsizei)(sizeof(fillIdx)/sizeof(fillIdx[0])), [](){
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
            });
        }
        
        // ============= Create Corner Marker VBO =============
        GLuint markerVAO=0, markerVBO=0;
        glGenVertexArrays(1,&markerVAO);
        glGenBuffers(1,&markerVBO);
        glBindVertexArray(markerVAO);
        glBindBuffer(GL_ARRAY_BUFFER, markerVBO);
        glBufferData(GL_ARRAY_BUFFER, 4 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
        glBindVertexArray(0);
        
        // ============= Create Corner Pins =============
        MeshRAII cornerPins;
        {
            float pinH = std::max(cubeHalf*0.8f, 10.f);
            float pinVerts[] = {
                0,0,0,   0,0,-pinH,
                0,0,0,   0,0,-pinH,
                0,0,0,   0,0,-pinH,
                0,0,0,   0,0,-pinH
            };
            cornerPins.initVertexOnly(pinVerts, sizeof(pinVerts), 8, [](){
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
            });
        }
        
        // Get uniform locations
        GLint marker_uMVP = markerProgram.getUniformLocation("uMVP");
        GLint marker_uColor = markerProgram.getUniformLocation("uColor");
        GLint marker_uPointSize = markerProgram.getUniformLocation("uPointSize");
        GLint plane_uMVP = planeProgram.getUniformLocation("uMVP");
        GLint plane_uColor = planeProgram.getUniformLocation("uPlaneColor");
        GLint planeLight_uMVP = planeLightProgram.getUniformLocation("uMVP");
        GLint planeLight_uMV = planeLightProgram.getUniformLocation("uMV");
        GLint planeLight_uColor = planeLightProgram.getUniformLocation("uColor");
        GLint planeLight_uHalfExtents = planeLightProgram.getUniformLocation("uHalfExtents");
        GLint sphere_uMVP = sphereProgram.getUniformLocation("uMVP");
        GLint sphere_uMV = sphereProgram.getUniformLocation("uMV");
        GLint sphere_uNormalMat = sphereProgram.getUniformLocation("uNormalMat");
        GLint sphere_uColor = sphereProgram.getUniformLocation("uColor");
        GLint sphere_uShininess = sphereProgram.getUniformLocation("uShininess");
        
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        // ============= Create Rendering Resources =============
        GLTexture bgTexture(width, height, frame.channels());
        MeshRAII bg = createBackgroundQuadRAII();
        float halfW = A4_w * 0.5f;
        float halfH = A4_h * 0.5f;
        float wallThickness = std::min(halfW, halfH) * 0.08f;
        std::vector<MeshRAII> walls = createWallsRAII(halfW, halfH, cubeH, wallThickness);
        std::vector<MeshRAII> mazeWalls = createMazeWallsRAII(halfW, halfH, cubeH, wallThickness);
        float sphereRadius = std::min(halfW, halfH) * 0.12f;
        MeshRAII sphere = createSphereRAII(sphereRadius, 36, 24);
        AxisMeshRAII axes = createAxesRAII(std::max(cubeHalf*1.5f, 30.0f));
        
        GLint bg_uTex = bgProgram.getUniformLocation("uTex");
        GLint line_uMVP = lineProgram.getUniformLocation("uMVP");
        GLint line_uColor = lineProgram.getUniformLocation("uColor");
        GLint line_uThickness = lineProgram.getUniformLocation("uThicknessPx");
        GLint line_uViewport = lineProgram.getUniformLocation("uViewport");
        GLint wall_uMVP = wallLightProgram.getUniformLocation("uMVP");
        GLint wall_uMV  = wallLightProgram.getUniformLocation("uMV");
        GLint wall_uColor = wallLightProgram.getUniformLocation("uColor");
        const float THICKNESS_PX = Config::render.lineThicknessPx;
        
        // ============= Initialize Tracking State =============
        TemporalValidator temporal_validator;
        MultiScaleDetector multi_scale_detector;
        RobustPoseEstimator pose_estimator;
        std::vector<cv::Point2f> lastCorners;
        cv::Mat prevGray;
        int totalFrames = 0, successfulDetections = 0, robustDetections = 0;
        bool hasPrevPose = false;
        cv::Mat rvecPrev, tvecPrev;
        const double poseAlpha = Config::pose.smoothingAlpha;
        const float qAlpha = (float)Config::pose.quadSmoothingAlpha;
        std::array<cv::Point2f,4> smoothQ;
        bool hadSmoothQuad = false;
        int frameNo = 0;
        
        // ============= Frame Processing Function =============
        auto processFrame = [&](cv::Mat& f) {
            if (f.empty()) return;
            if (!isImage && Config::runtime.showPoseMetrics) totalFrames++;
            
            cv::Mat uFrame;
            cv::remap(f, uFrame, undistortMap1, undistortMap2, cv::INTER_LINEAR);
            
            int fbw, fbh;
            glfw_manager.getFramebufferSize(fbw, fbh);
            glViewport(0, 0, fbw, fbh);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            cv::Mat gray;
            cv::cvtColor(uFrame, gray, cv::COLOR_BGR2GRAY);
            
            // Detection
            std::vector<cv::Point2f> quad;
            bool detected = detectA4CornersRobust(uFrame, quad, temporal_validator, multi_scale_detector);
            bool usedTracking = false;
            
            if (!detected && !lastCorners.empty() && !prevGray.empty()) {
                std::vector<uchar> status;
                std::vector<float> err;
                std::vector<cv::Point2f> tracked;
                cv::calcOpticalFlowPyrLK(prevGray, gray, lastCorners, tracked, status, err,
                                         cv::Size(21,21), 4,
                                         cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 50, 0.01),
                                         cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 1e-5);
                int okCount=0;
                std::vector<cv::Point2f> valid_prev, valid_trk;
                for(size_t i=0;i<status.size();++i){
                    if(status[i] && err[i]<20.0 &&
                       tracked[i].x>=0 && tracked[i].x<uFrame.cols &&
                       tracked[i].y>=0 && tracked[i].y<uFrame.rows){
                        valid_prev.push_back(lastCorners[i]);
                        valid_trk.push_back(tracked[i]);
                        okCount++;
                    }
                }
                if(okCount>=4){
                    cv::Mat H = cv::findHomography(valid_prev, valid_trk, cv::RANSAC, 3.0);
                    if(!H.empty() && std::abs(cv::determinant(H))>1e-6){
                        quad = valid_trk;
                        usedTracking = true;
                        temporal_validator.addDetection(quad, 0.3);
                    }
                } else if (temporal_validator.isValid() && temporal_validator.predicted_corners.size()==4){
                    quad = temporal_validator.predicted_corners;
                    usedTracking = true;
                    temporal_validator.addDetection(quad, 0.25);
                }
            }
            
            // Update background texture
            cv::Mat texImg, flipped;
            if (uFrame.channels()==3) cv::cvtColor(uFrame, texImg, cv::COLOR_BGR2RGB);
            else if (uFrame.channels()==4) cv::cvtColor(uFrame, texImg, cv::COLOR_BGRA2RGBA);
            else texImg = uFrame;
            cv::flip(texImg, flipped, 0);
            bgTexture.updateFromMat(flipped);
            
            // Draw background
            glDisable(GL_DEPTH_TEST);
            bgProgram.use();
            bgTexture.bind(0);
            glUniform1i(bg_uTex, 0);
            bg.bind();
            glDrawArrays(GL_TRIANGLES, 0, bg.count());
            bg.unbind();
            bgTexture.unbind();
            glEnable(GL_DEPTH_TEST);
            
            if (detected || usedTracking) {
                if(detected) successfulDetections++;
                if(detected && temporal_validator.getCurrentConfidence()>0.7) robustDetections++;
                
                std::vector<cv::Point2f> quadOrdered = orderQuadRobust(quad);
                if(quadOrdered.size()!=4){
                    prevGray = gray.clone();
                    frameNo++;
                    return;
                }
                
                // Smooth corners
                double conf = temporal_validator.getCurrentConfidence();
                float adaptive_alpha = qAlpha * (1.f - (float)conf * 0.3f);
                std::array<cv::Point2f,4> quadSm;
                if(!hadSmoothQuad){
                    for(int i=0;i<4;i++){ smoothQ[i]=quadOrdered[i]; quadSm[i]=quadOrdered[i]; }
                    hadSmoothQuad=true;
                } else {
                    for(int i=0;i<4;i++){
                        smoothQ[i] = adaptive_alpha * smoothQ[i] + (1.f - adaptive_alpha) * quadOrdered[i];
                        quadSm[i] = smoothQ[i];
                    }
                }
                lastCorners.assign(quadSm.begin(), quadSm.end());
                prevGray = gray.clone();
                
                // 2D corner visualization
                cv::Scalar dotColor = detected ?
                    (conf>0.8 ? cv::Scalar(0,255,0) : cv::Scalar(0,165,255)) :
                    cv::Scalar(255,0,0);
                int dotSize = (int)(4 + conf*4);
                for (auto& c : quadSm) cv::circle(uFrame, c, dotSize, dotColor, -1);
                std::vector<cv::Point> poly; for(auto &c:quadSm) poly.emplace_back(cvRound(c.x),cvRound(c.y));
                cv::polylines(uFrame, std::vector<std::vector<cv::Point>>{poly}, true, cv::Scalar(0,255,0), 2);
                
                // Pose estimation
                cv::Mat rvec, tvec;
                double pose_conf = pose_estimator.estimateRobustPose(objectPoints,
                    {quadSm[0],quadSm[1],quadSm[2],quadSm[3]}, cameraMatrix, distCoeffs, rvec, tvec);
                if(pose_conf<0){
                    frameNo++;
                    return;
                }
                
                // Pose smoothing
                cv::Mat rdisp, tdisp;
                if(hasPrevPose && pose_conf>0.5){
                    Quat qPrev=rotvecToQuat(rvecPrev), qCurr=rotvecToQuat(rvec);
                    double adaptive_pose_alpha = poseAlpha * (2.0 - pose_conf);
                    adaptive_pose_alpha = std::min(0.8, std::max(0.1, adaptive_pose_alpha));
                    Quat qMix = quatSlerp(qPrev, qCurr, adaptive_pose_alpha);
                    rdisp = quatToRotvec(qMix);
                    tdisp = tvecPrev*(1.0-adaptive_pose_alpha) + tvec*adaptive_pose_alpha;
                } else {
                    rdisp = rvec.clone(); tdisp = tvec.clone(); hasPrevPose=true;
                }
                rvecPrev = rdisp.clone(); tvecPrev = tdisp.clone();
                
                // Projection / view
                glm::mat4 projection = projectionFromCV(cameraMatrix, (float)uFrame.cols, (float)uFrame.rows, 0.1f, 3000.f);
                glm::mat4 view = getModelViewMatrix(rdisp, tdisp);
                glm::mat4 model(1.f);
                glm::mat4 MVP = projection * view * model;
                
                // Update marker VBO
                float markerData[12] = {
                    objectPoints[0].x, objectPoints[0].y, 0.f,
                    objectPoints[1].x, objectPoints[1].y, 0.f,
                    objectPoints[2].x, objectPoints[2].y, 0.f,
                    objectPoints[3].x, objectPoints[3].y, 0.f
                };
                glBindBuffer(GL_ARRAY_BUFFER, markerVBO);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(markerData), markerData);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                
                float sphereRadius = 0.12f * std::min(halfW, halfH);
                glm::vec2 ballPos(0.f, 0.f);
                
                // Plane fill with lighting
                planeLightProgram.use();
                glUniformMatrix4fv(planeLight_uMVP, 1, GL_FALSE, glm::value_ptr(MVP));
                glm::mat4 planeMV = view;
                glUniformMatrix4fv(planeLight_uMV, 1, GL_FALSE, glm::value_ptr(planeMV));
                glUniform2f(planeLight_uHalfExtents, halfW, halfH);
                float planeAlpha = 0.15f + 0.35f * (float)pose_conf;
                glUniform4f(planeLight_uColor, 0.1f,0.3f,0.9f,planeAlpha);
                planeFill.bind(); glDrawElements(GL_TRIANGLES, planeFill.count(), GL_UNSIGNED_INT, 0); planeFill.unbind();
                
                planeProgram.use();
                glUniformMatrix4fv(plane_uMVP, 1, GL_FALSE, glm::value_ptr(MVP));
                glUniform4f(plane_uColor, 0.8f,0.85f,0.9f,0.9f);
                planeWire.bind(); glDrawElements(GL_LINES, planeWire.count(), GL_UNSIGNED_INT, 0); planeWire.unbind();
                
                // Sphere
                glm::mat4 sphereModel = glm::translate(glm::mat4(1.f), glm::vec3(ballPos.x, ballPos.y, -sphereRadius));
                glm::mat4 sphereMV = view * sphereModel;
                glm::mat4 sphereMVP = projection * sphereMV;
                glm::mat3 sphereNormalMat = glm::transpose(glm::inverse(glm::mat3(sphereMV)));
                sphereProgram.use();
                glUniformMatrix4fv(sphere_uMVP, 1, GL_FALSE, glm::value_ptr(sphereMVP));
                glUniformMatrix4fv(sphere_uMV, 1, GL_FALSE, glm::value_ptr(sphereMV));
                glUniformMatrix3fv(sphere_uNormalMat, 1, GL_FALSE, glm::value_ptr(sphereNormalMat));
                glUniform4f(sphere_uColor, 0.2f, 0.4f, 0.9f, 1.f);  // Blue sphere
                glUniform1f(sphere_uShininess, 32.f);
                sphere.bind(); glDrawElements(GL_TRIANGLES, sphere.count(), GL_UNSIGNED_INT, 0); sphere.unbind();
                
                // Corner pins
                lineProgram.use();
                glUniform2f(line_uViewport, (float)fbw, (float)fbh);
                glUniform1f(line_uThickness, THICKNESS_PX * ((float)fbw / (float)uFrame.cols));
                auto drawPin=[&](const cv::Point3f& p,const glm::vec3& col){
                    glm::mat4 modelPin = glm::translate(glm::mat4(1.f), glm::vec3(p.x,p.y,0.f));
                    glm::mat4 pinMVP = projection*view*modelPin;
                    glUniformMatrix4fv(line_uMVP,1,GL_FALSE,glm::value_ptr(pinMVP));
                    glUniform3f(line_uColor,col.r,col.g,col.b);
                    cornerPins.bind(); glDrawArrays(GL_LINES,0,2); cornerPins.unbind();
                };
                drawPin(objectPoints[0], {1.f,0.f,0.f});
                drawPin(objectPoints[1], {0.f,1.f,0.f});
                drawPin(objectPoints[2], {0.f,0.4f,1.f});
                drawPin(objectPoints[3], {1.f,0.85f,0.f});
                
                // Corner point sprites
                markerProgram.use();
                glUniformMatrix4fv(marker_uMVP, 1, GL_FALSE, glm::value_ptr(MVP));
                float ptSize = 10.f + 12.f*(float)pose_conf;
                glUniform1f(marker_uPointSize, ptSize);
                glBindVertexArray(markerVAO);
                auto setColor=[&](int i){
                    switch(i){
                        case 0: glUniform3f(marker_uColor,1.f,0.f,0.f); break;
                        case 1: glUniform3f(marker_uColor,0.f,1.f,0.f); break;
                        case 2: glUniform3f(marker_uColor,0.f,0.4f,1.f); break;
                        case 3: glUniform3f(marker_uColor,1.f,0.85f,0.f); break;
                    }
                    glDrawArrays(GL_POINTS,i,1);
                };
                for(int i=0;i<4;i++) setColor(i);
                glBindVertexArray(0);
                
                // Walls
                wallLightProgram.use();
                for(size_t i=0; i<walls.size(); i++){
                    glm::mat4 wallMVP = projection * view;
                    glm::mat4 wallMV = view;
                    glUniformMatrix4fv(wall_uMVP, 1, GL_FALSE, glm::value_ptr(wallMVP));
                    glUniformMatrix4fv(wall_uMV, 1, GL_FALSE, glm::value_ptr(wallMV));
                    glUniform4f(wall_uColor, 0.75f, 0.75f, 0.75f, 0.9f);  // Light gray
                    walls[i].bind(); glDrawElements(GL_TRIANGLES, walls[i].count(), GL_UNSIGNED_INT, 0); walls[i].unbind();
                }
                
                // Maze walls
                for(size_t i=0; i<mazeWalls.size(); i++){
                    glm::mat4 mazeMVP = projection * view;
                    glm::mat4 mazeMV = view;
                    glUniformMatrix4fv(wall_uMVP, 1, GL_FALSE, glm::value_ptr(mazeMVP));
                    glUniformMatrix4fv(wall_uMV, 1, GL_FALSE, glm::value_ptr(mazeMV));
                    glUniform4f(wall_uColor, 0.75f, 0.75f, 0.75f, 0.9f);  // Light gray
                    mazeWalls[i].bind(); glDrawElements(GL_TRIANGLES, mazeWalls[i].count(), GL_UNSIGNED_INT, 0); mazeWalls[i].unbind();
                }
                
                // Axes
                lineProgram.use();
                glUniformMatrix4fv(line_uMVP,1,GL_FALSE,glm::value_ptr(MVP));
                axes.bind();
                glUniform3f(line_uColor,1.f,0.f,0.f); glDrawArrays(GL_LINES,0,2);
                glUniform3f(line_uColor,0.f,1.f,0.f); glDrawArrays(GL_LINES,2,2);
                glUniform3f(line_uColor,0.f,0.f,1.f); glDrawArrays(GL_LINES,4,2);
                axes.unbind();
            }
            
            frameNo++;
            prevGray = gray.clone();
            if(!isImage && Config::runtime.showPoseMetrics &&
               (frameNo % Config::runtime.poseMetricsInterval)==0){
                double hitRate = (double)successfulDetections / std::max(1,totalFrames) * 100.0;
                double robustRate = (double)robustDetections / std::max(1,totalFrames) * 100.0;
                std::cout << "Stats | Success: " << successfulDetections << "/" << totalFrames
                          << " (" << std::fixed << std::setprecision(1) << hitRate << "%)"
                          << " | Robust: " << std::setprecision(1) << robustRate << "%\n";
            }
        };
        
        // ============= Main Loop =============
        processFrame(frame);
        
        if (isImage) {
            cv::Mat base = frame.clone();
            while (!glfw_manager.shouldClose()) {
                cv::Mat display = base.clone();
                processFrame(display);
                glfw_manager.swapBuffers();
                glfw_manager.pollEvents();
            }
        } else {
            VideoCaptureRAII capture(source);
            if (!capture.isOpened()) {
                throw std::runtime_error("Failed to open video source");
            }
            
            double fps = capture.get(cv::CAP_PROP_FPS);
            if (fps <= 1.0 || fps > 120.0) fps = 30.0;
            int delayMs = static_cast<int>(1000.0 / fps);
            
            while (!glfw_manager.shouldClose() && capture.read(frame)) {
                double t0 = cv::getTickCount();
                processFrame(frame);
                glfw_manager.swapBuffers();
                glfw_manager.pollEvents();
                
                int key = cv::waitKey(1);
                if (key == 27) glfwSetWindowShouldClose(window, GLFW_TRUE);
                
                double t1 = cv::getTickCount();
                double elapsedMs = (t1 - t0) * 1000.0 / cv::getTickFrequency();
                int waitMs = delayMs - static_cast<int>(elapsedMs);
                if (waitMs > 0) {
                    int key2 = cv::waitKey(waitMs);
                    if (key2 == 27) glfwSetWindowShouldClose(window, GLFW_TRUE);
                }
            }
        }
        
        // ============= Final Statistics =============
        if (totalFrames > 0) {
            double finalHitRate = (double)successfulDetections / totalFrames * 100.0;
            double robustRate = (double)robustDetections / totalFrames * 100.0;
            double avgConfidence = temporal_validator.getCurrentConfidence();
            
            std::cout << "\n=== ENHANCED DETECTION SUMMARY ===\n"
                      << "Total frames: " << totalFrames << "\n"
                      << "Successful detections: " << successfulDetections 
                      << " (" << std::fixed << std::setprecision(2) << finalHitRate << "%)\n"
                      << "Robust detections: " << robustDetections 
                      << " (" << std::setprecision(2) << robustRate << "%)\n"
                      << "Average confidence: " << std::setprecision(2) << avgConfidence << "\n"
                      << "Detection quality: " << (robustRate > 70 ? "EXCELLENT" : robustRate > 50 ? "GOOD" : "FAIR") << "\n";
        }
        
        std::cout << "Application completed successfully. All resources cleaned up.\n";
        return 0;
        
    } catch (const ValidationException& ve) {
        std::cerr << ve.what() << "\nTerminating due to validation failure.\n";
        return -2;
    } catch (const ResourceException& re) {
        std::cerr << re.what() << "\nTerminating due to resource allocation failure.\n";
        return -3;
    } catch (const ProcessingException& pe) {
        std::cerr << pe.what() << "\nTerminating due to processing failure.\n";
        return -4;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << "\n";
        return -1;
    }
}
