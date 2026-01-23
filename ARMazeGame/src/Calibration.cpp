#include "Calibration.hpp"
#include <filesystem>
#include <iostream>
#include <vector>
#include <cmath>

static const cv::Size kBoardSize(9, 6);
static const float kSquareSize = 0.024f;  // 24mm squares

bool runLiveCalibration(int camIndex,
                        const std::string& yamlPath,
                        cv::Size& calibSize_out,
                        cv::Mat& K_out, cv::Mat& D_out) {
    cv::VideoCapture cap(camIndex);
    if (!cap.isOpened()) {
        std::cerr << "Calibration: Cannot open webcam index " << camIndex << "\n";
        return false;
    }

    std::vector<std::vector<cv::Point2f>> imgPoints;
    std::vector<std::vector<cv::Point3f>> objPoints;

    // Create object points for chessboard
    std::vector<cv::Point3f> obj;
    obj.reserve(kBoardSize.area());
    for (int j = 0; j < kBoardSize.height; ++j) {
        for (int i = 0; i < kBoardSize.width; ++i) {
            obj.emplace_back(i * kSquareSize, j * kSquareSize, 0.f);
        }
    }

    std::cout << "[Calibration]\n"
              << " - Show the checkerboard pattern (" << kBoardSize.width << "x" << kBoardSize.height << ")\n"
              << " - Press 'c' to CAPTURE when the pattern is detected\n"
              << " - Press 'a' to toggle AUTO capture mode\n"
              << " - Press 'q' to QUIT\n";

    const int needed = 20;
    cv::Mat frame, gray;

    // Helper functions for auto-capture logic
    auto areaOf = [](const std::vector<cv::Point2f>& c) {
        cv::Rect bb = cv::boundingRect(c);
        return (double)bb.area();
    };
    
    auto rmsShift = [](const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
        if (a.size() != b.size() || a.empty()) return 1e9;
        double s = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            s += cv::norm(a[i] - b[i]) * cv::norm(a[i] - b[i]);
        }
        return std::sqrt(s / a.size());
    };
    
    bool autoMode = true;
    std::vector<cv::Point2f> lastCaptured;
    double lastArea = 0.0;
    int64_t lastTick = cv::getTickCount();
    const double tick2ms = 1000.0 / cv::getTickFrequency();

    while ((int)imgPoints.size() < needed) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Calibration: Empty frame\n";
            break;
        }
        
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(
            gray, kBoardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
        );

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.01));
            cv::drawChessboardCorners(frame, kBoardSize, corners, true);
            
            std::string msg = autoMode ? "Pattern detected - AUTO capture" : "Pattern detected - 'c' to capture";
            cv::putText(frame, msg, {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);

            if (autoMode) {
                int64_t now = cv::getTickCount();
                double dt_ms = (now - lastTick) * tick2ms;

                double curArea = areaOf(corners);
                double areaRelChange = (lastArea > 0.0) ? std::abs(curArea - lastArea) / lastArea : 1.0;
                double move = lastCaptured.empty() ? 1e9 : rmsShift(corners, lastCaptured);

                bool enoughChange = (move > 15.0) || (areaRelChange > 0.12);
                bool enoughTime = (dt_ms > 700.0);

                if (enoughTime && enoughChange) {
                    imgPoints.push_back(corners);
                    objPoints.push_back(obj);

                    lastCaptured = corners;
                    lastArea = curArea;
                    lastTick = now;
                    
                    std::cout << "Auto-captured view " << imgPoints.size() << "/" << needed << "\n";
                }
            }
        } else {
            cv::putText(frame, "Looking for pattern...", {20, 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 255}, 2);
        }

        // Draw status
        std::ostringstream ss;
        ss << "Views: " << imgPoints.size() << "/" << needed
           << "   [a] Auto:" << (autoMode ? "ON" : "OFF")
           << "   [c] Capture   [q] Quit";
        cv::putText(frame, ss.str(), {20, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 2);

        // Progress bar
        int progress = (int)(200.0 * imgPoints.size() / needed);
        cv::rectangle(frame, cv::Rect(20, 70, 200, 15), cv::Scalar(50, 50, 50), -1);
        cv::rectangle(frame, cv::Rect(20, 70, progress, 15), cv::Scalar(0, 255, 0), -1);

        cv::imshow("Calibration", frame);
        int k = cv::waitKey(1);
        
        if (k == 'q' || k == 27) {
            cv::destroyWindow("Calibration");
            return false;
        }
        if (k == 'a') {
            autoMode = !autoMode;
        }
        if (k == 'c' && found) {
            imgPoints.push_back(corners);
            objPoints.push_back(obj);
            lastCaptured = corners;
            lastArea = areaOf(corners);
            lastTick = cv::getTickCount();
            std::cout << "Manually captured view " << imgPoints.size() << "/" << needed << "\n";
        }
    }
    cv::destroyWindow("Calibration");

    if ((int)imgPoints.size() < needed) {
        std::cerr << "Calibration: Not enough images captured\n";
        return false;
    }

    // Perform calibration
    cv::Mat K, D;
    std::vector<cv::Mat> rvecs, tvecs;
    calibSize_out = frame.size();
    
    double rms = cv::calibrateCamera(objPoints, imgPoints, calibSize_out, K, D, rvecs, tvecs,
                                     cv::CALIB_RATIONAL_MODEL);
    std::cout << "Calibration complete. RMS = " << rms << "\n";

    // Save calibration
    std::filesystem::create_directories(std::filesystem::path(yamlPath).parent_path());
    cv::FileStorage fs(yamlPath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Calibration: Cannot write YAML: " << yamlPath << "\n";
        return false;
    }
    
    fs << "image_width" << calibSize_out.width;
    fs << "image_height" << calibSize_out.height;
    fs << "camera_matrix" << K;
    fs << "distortion_coefficients" << D;
    fs.release();

    std::cout << "Calibration saved to: " << yamlPath << "\n";

    K.convertTo(K_out, CV_64F);
    D.convertTo(D_out, CV_64F);

    // Show completion message
    cv::putText(frame, "Calibration complete!", {20, 40},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
    cv::imshow("Calibration", frame);
    cv::waitKey(1000);

    return true;
}

void createDefaultCalibration(int width, int height, cv::Mat& K, cv::Mat& D) {
    // Estimate focal length based on image size and typical webcam FOV
    double diagonalFovDeg = 60.0;  // Typical webcam FOV
    double diagonalPixels = std::sqrt((double)width * width + (double)height * height);
    double f = diagonalPixels / (2.0 * std::tan(diagonalFovDeg * CV_PI / 360.0));
    
    // Construct intrinsic matrix
    K = (cv::Mat_<double>(3, 3) <<
        f, 0, width * 0.5,
        0, f, height * 0.5,
        0, 0, 1);
    
    // Zero distortion
    D = cv::Mat::zeros(1, 5, CV_64F);
    
    std::cout << "Created default calibration for " << width << "x" << height << "\n";
    std::cout << "Estimated focal length: " << f << "\n";
}
