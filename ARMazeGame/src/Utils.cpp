#include "Utils.hpp"
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <fstream>

std::filesystem::path resolveDataRoot(const std::filesystem::path& exePath) {
    // Check environment variable first
    if (const char* env = std::getenv("AR_DATA_DIR")) {
        std::filesystem::path p(env);
        if (std::filesystem::exists(p)) {
            std::cout << "Using AR_DATA_DIR: " << p << "\n";
            return std::filesystem::canonical(p);
        }
    }

    std::filesystem::path exeDir;
    try {
        exeDir = std::filesystem::canonical(exePath).parent_path();
    } catch (...) {
        exeDir = exePath.parent_path();
        if (exeDir.empty()) {
            exeDir = std::filesystem::current_path();
        }
    }
    
    std::cout << "Executable directory: " << exeDir << "\n";

    // Common locations relative to executable (ordered by priority)
    std::vector<std::filesystem::path> candidates = {
        exeDir / "data",                              // build/data (copied by cmake)
        exeDir / ".." / "data",                       // ARMazeGame/data (when running from build/)
        exeDir / ".." / ".." / "data",                // parent's parent
        std::filesystem::current_path() / "data",     // current directory
        std::filesystem::current_path() / ".." / "data"
    };

    std::cout << "Searching for data directory...\n";
    for (auto& c : candidates) {
        std::error_code ec;
        std::filesystem::path normalized = std::filesystem::weakly_canonical(c, ec);
        if (!ec && std::filesystem::exists(normalized, ec) && std::filesystem::is_directory(normalized, ec)) {
            std::cout << "Found data at: " << normalized << "\n";
            return normalized;
        }
    }

    // Fallback: create data directory in current working directory
    std::filesystem::path fallback = std::filesystem::current_path() / "data";
    std::cout << "Data directory not found, using: " << fallback << "\n";
    return fallback;
}

bool loadCalibration(const std::string& yamlPath, cv::Mat& K, cv::Mat& D, cv::Size& calibSize) {
    cv::FileStorage fs(yamlPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Cannot open calibration file: " << yamlPath << "\n";
        return false;
    }
    
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> D;
    
    int w = 0, h = 0;
    if (fs["image_width"].isInt())  w = (int)fs["image_width"];
    if (fs["image_height"].isInt()) h = (int)fs["image_height"];
    calibSize = (w > 0 && h > 0) ? cv::Size(w, h) : cv::Size();

    if (K.empty()) {
        std::cerr << "Error: camera_matrix not found in " << yamlPath << "\n";
        return false;
    }
    
    if (D.empty()) {
        D = cv::Mat::zeros(1, 5, CV_64F);
    }
    
    K.convertTo(K, CV_64F);
    D.convertTo(D, CV_64F);
    
    return true;
}

cv::Mat scaleIntrinsics(const cv::Mat& K0, cv::Size from, cv::Size to) {
    if (from.width <= 0 || from.height <= 0 || from == to) {
        return K0.clone();
    }
    
    double sx = (double)to.width / (double)from.width;
    double sy = (double)to.height / (double)from.height;
    
    cv::Mat K1 = K0.clone();
    K1.at<double>(0, 0) *= sx; // fx
    K1.at<double>(1, 1) *= sy; // fy
    K1.at<double>(0, 2) *= sx; // cx
    K1.at<double>(1, 2) *= sy; // cy
    
    return K1;
}

void rotateFrameInPlace(cv::Mat& img, Rot r) {
    if (r == Rot::CW) {
        cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
    } else if (r == Rot::CCW) {
        cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
}

Rot decideRotationFromSizes(const cv::Size& calibSize, const cv::Size& frameSize) {
    if (calibSize.width <= 0 || calibSize.height <= 0) {
        return Rot::NONE;
    }
    
    bool calibPortrait = calibSize.height > calibSize.width;
    bool framePortrait = frameSize.height > frameSize.width;

    if (calibPortrait == framePortrait) {
        return Rot::NONE;
    }
    return Rot::CCW;
}

std::vector<std::string> listFilesInDirectory(const std::filesystem::path& dir, 
                                               const std::vector<std::string>& extensions) {
    std::vector<std::string> result;
    
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
        std::cout << "  Directory does not exist: " << dir << "\n";
        return result;
    }
    
    if (!std::filesystem::is_directory(dir, ec)) {
        std::cout << "  Not a directory: " << dir << "\n";
        return result;
    }
    
    std::cout << "  Scanning: " << dir << "\n";
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (ec) {
                std::cout << "  Error iterating: " << ec.message() << "\n";
                break;
            }
            
            if (entry.is_regular_file(ec)) {
                std::string filename = entry.path().filename().string();
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                for (const auto& validExt : extensions) {
                    std::string lowerValidExt = validExt;
                    std::transform(lowerValidExt.begin(), lowerValidExt.end(), 
                                 lowerValidExt.begin(), ::tolower);
                    if (ext == lowerValidExt) {
                        std::cout << "    Found: " << filename << "\n";
                        result.push_back(filename);
                        break;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cout << "  Exception scanning directory: " << e.what() << "\n";
    }
    
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<int> getAvailableWebcams(int maxCheck) {
    std::vector<int> result;
    
    for (int i = 0; i < maxCheck; ++i) {
        try {
            // First check if the device exists
            std::string devPath = "/dev/video" + std::to_string(i);
            std::ifstream devCheck(devPath);
            if (!devCheck.good()) {
                continue;  // Device doesn't exist, skip
            }
            devCheck.close();
            
            cv::VideoCapture cap;
            // Use V4L2 backend explicitly on Linux to reduce warnings
            #ifdef __linux__
            cap.open(i, cv::CAP_V4L2);
            #else
            cap.open(i);
            #endif
            
            if (cap.isOpened()) {
                cv::Mat testFrame;
                // Try to read a frame to confirm it works
                bool gotFrame = cap.read(testFrame);
                cap.release();
                
                if (gotFrame && !testFrame.empty()) {
                    result.push_back(i);
                }
            }
        } catch (...) {
            // Silently ignore any errors
            continue;
        }
    }
    
    return result;
}