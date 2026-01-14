/**
 * @file Calibration.cpp
 * @brief Implementation of camera calibration management
 */

#include "Calibration.hpp"
#include "Config.hpp"
#include "Exceptions.hpp"
#include <iostream>
#include <iomanip>
#include <ctime>

// Global variables
cv::Mat cameraMatrix, distCoeffs;
CameraCalibration current_calibration;

// ============= CameraCalibration =============
CameraCalibration::CameraCalibration() 
    : reprojection_error(0.0), is_valid(false) {}

bool CameraCalibration::validateCameraMatrix() const {
    if (camera_matrix.rows != 3 || camera_matrix.cols != 3) return false;
    if (camera_matrix.type() != CV_64F) return false;
    
    double fx = camera_matrix.at<double>(0, 0);
    double fy = camera_matrix.at<double>(1, 1);
    double cx = camera_matrix.at<double>(0, 2);
    double cy = camera_matrix.at<double>(1, 2);
    
    return (fx > 0 && fy > 0 && cx > 0 && cy > 0 && 
            fx < 10000 && fy < 10000 &&
            cx < image_size.width && cy < image_size.height);
}

bool CameraCalibration::validateDistortionCoeffs() const {
    if (distortion_coeffs.empty()) return true;
    if (distortion_coeffs.type() != CV_64F) return false;
    
    for (int i = 0; i < distortion_coeffs.total(); ++i) {
        double coeff = distortion_coeffs.at<double>(i);
        if (std::abs(coeff) > 10.0) return false;
    }
    return true;
}

CameraCalibration CameraCalibration::scaleToSize(const cv::Size& new_size) const {
    if (!is_valid || image_size.width <= 0 || image_size.height <= 0) {
        return *this;
    }
    
    CameraCalibration scaled = *this;
    
    double scale_x = static_cast<double>(new_size.width) / image_size.width;
    double scale_y = static_cast<double>(new_size.height) / image_size.height;
    
    scaled.camera_matrix = camera_matrix.clone();
    scaled.camera_matrix.at<double>(0, 0) *= scale_x; // fx
    scaled.camera_matrix.at<double>(0, 2) *= scale_x; // cx
    scaled.camera_matrix.at<double>(1, 1) *= scale_y; // fy
    scaled.camera_matrix.at<double>(1, 2) *= scale_y; // cy
    
    scaled.distortion_coeffs = distortion_coeffs.clone();
    scaled.image_size = new_size;
    
    return scaled;
}

std::pair<double, double> CameraCalibration::getFieldOfView() const {
    if (!is_valid) return {0.0, 0.0};
    
    double fx = camera_matrix.at<double>(0, 0);
    double fy = camera_matrix.at<double>(1, 1);
    
    double hfov = 2.0 * std::atan(image_size.width / (2.0 * fx)) * 180.0 / CV_PI;
    double vfov = 2.0 * std::atan(image_size.height / (2.0 * fy)) * 180.0 / CV_PI;
    
    return {hfov, vfov};
}

void CameraCalibration::printSummary() const {
    std::cout << "\n=== CAMERA CALIBRATION SUMMARY ===\n";
    std::cout << "Method: " << calibration_method << "\n";
    std::cout << "Image size: " << image_size.width << "x" << image_size.height << "\n";
    std::cout << "RMS error: " << std::fixed << std::setprecision(3) << reprojection_error << " pixels\n";
    
    if (is_valid) {
        double fx = camera_matrix.at<double>(0, 0);
        double fy = camera_matrix.at<double>(1, 1);
        double cx = camera_matrix.at<double>(0, 2);
        double cy = camera_matrix.at<double>(1, 2);
        
        std::cout << "Focal lengths: fx=" << std::setprecision(2) << fx << ", fy=" << fy << "\n";
        std::cout << "Principal point: cx=" << std::setprecision(2) << cx << ", cy=" << cy << "\n";
        
        auto [hfov, vfov] = getFieldOfView();
        std::cout << "Field of view: " << std::setprecision(1) << hfov << "° x " << vfov << "°\n";
        
        std::cout << "Distortion coefficients: ";
        if (distortion_coeffs.total() > 0) {
            for (int i = 0; i < std::min(5, static_cast<int>(distortion_coeffs.total())); ++i) {
                std::cout << std::setprecision(6) << distortion_coeffs.at<double>(i);
                if (i < 4 && i < distortion_coeffs.total() - 1) std::cout << ", ";
            }
        } else {
            std::cout << "None (zero distortion)";
        }
        std::cout << "\n";
    }
    std::cout << "Status: " << (is_valid ? "VALID" : "INVALID") << "\n";
    std::cout << "================================\n\n";
}

// ============= Functions =============
void buildDefaultCalibration(int width, int height) {
    double w = static_cast<double>(width);
    double h = static_cast<double>(height);
    
    double diagonal_fov_deg = 60.0;
    double diagonal_pixels = std::sqrt(w * w + h * h);
    double f_diagonal = diagonal_pixels / (2.0 * std::tan(diagonal_fov_deg * CV_PI / 360.0));
    double fx = f_diagonal * std::sqrt(w * w) / diagonal_pixels;
    double fy = fx;
    
    current_calibration.camera_matrix = (cv::Mat_<double>(3, 3) << 
        fx, 0, w * 0.5,
        0, fy, h * 0.5,
        0, 0, 1);
    
    current_calibration.distortion_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    current_calibration.image_size = cv::Size(width, height);
    current_calibration.reprojection_error = 0.0;
    current_calibration.calibration_method = "Default (estimated)";
    current_calibration.calibration_date = std::chrono::system_clock::now();
    current_calibration.is_valid = true;
    
    cameraMatrix = current_calibration.camera_matrix.clone();
    distCoeffs = current_calibration.distortion_coeffs.clone();
    
    std::cout << "Using default calibration parameters:\n";
    current_calibration.printSummary();
}

void loadCalibration(const std::string& filename, int width, int height) {
    if (filename.empty()) {
        buildDefaultCalibration(width, height);
        return;
    }
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Calibration YAML not found ('" << filename << "'). Using default parameters.\n";
        buildDefaultCalibration(width, height);
        return;
    }
    
    current_calibration = CameraCalibration();
    
    fs["camera_matrix"] >> current_calibration.camera_matrix;
    fs["distortion_coefficients"] >> current_calibration.distortion_coeffs;
    
    if (!fs["calibration_method"].empty()) {
        fs["calibration_method"] >> current_calibration.calibration_method;
    } else {
        current_calibration.calibration_method = "Loaded from file";
    }
    
    if (!fs["reprojection_error"].empty()) {
        fs["reprojection_error"] >> current_calibration.reprojection_error;
    }
    
    int orig_w = 0, orig_h = 0;
    if (!fs["image_width"].empty()) fs["image_width"] >> orig_w;
    if (!fs["image_height"].empty()) fs["image_height"] >> orig_h;
    
    if (orig_w > 0 && orig_h > 0) {
        current_calibration.image_size = cv::Size(orig_w, orig_h);
    } else {
        current_calibration.image_size = cv::Size(width, height);
        std::cout << "Warning: Original calibration image size not found. Assuming current size.\n";
    }
    
    if (!current_calibration.camera_matrix.empty()) {
        current_calibration.camera_matrix.convertTo(current_calibration.camera_matrix, CV_64F);
    }
    if (!current_calibration.distortion_coeffs.empty()) {
        current_calibration.distortion_coeffs.convertTo(current_calibration.distortion_coeffs, CV_64F);
    } else {
        current_calibration.distortion_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    }
    
    current_calibration.is_valid = current_calibration.validateCameraMatrix() && 
                                   current_calibration.validateDistortionCoeffs();
    
    if (!current_calibration.is_valid) {
        std::cerr << "Warning: Invalid calibration matrix/disto. Falling back to defaults.\n";
        buildDefaultCalibration(width, height);
        return;
    }
    
    try {
        require(current_calibration.camera_matrix.type() == CV_64F, "camera_matrix must be CV_64F");
        require(current_calibration.camera_matrix.rows == 3 && current_calibration.camera_matrix.cols == 3, 
                "camera_matrix must be 3x3");
        require(current_calibration.camera_matrix.at<double>(0, 0) > 0, "fx must be > 0");
        require(current_calibration.camera_matrix.at<double>(1, 1) > 0, "fy must be > 0");
    } catch (const ValidationException& ve) {
        std::cerr << ve.what() << "\nReverting to default calibration.\n";
        buildDefaultCalibration(width, height);
        return;
    }
    
    cv::Size target_size(width, height);
    if (current_calibration.image_size != target_size) {
        std::cout << "Scaling calibration from " << current_calibration.image_size.width 
                  << "x" << current_calibration.image_size.height << " to " 
                  << width << "x" << height << "\n";
        current_calibration = current_calibration.scaleToSize(target_size);
    }
    
    cameraMatrix = current_calibration.camera_matrix.clone();
    distCoeffs = current_calibration.distortion_coeffs.clone();
    
    std::cout << "Loaded calibration from: " << filename << "\n";
    current_calibration.printSummary();
    
    auto [hfov, vfov] = current_calibration.getFieldOfView();
    if (hfov < 10 || hfov > 180 || vfov < 10 || vfov > 180) {
        std::cout << "Warning: Unusual field of view detected. Please verify calibration.\n";
    }
    
    if (current_calibration.reprojection_error > 2.0) {
        std::cout << "Warning: High reprojection error. Consider recalibrating camera.\n";
    }
}

CameraCalibration performAutoCalibration(const std::vector<cv::Mat>& images,
                                         const cv::Size& board_size,
                                         float square_size)
{
    CameraCalibration calib;
    if (images.empty()) {
        std::cerr << "No images provided for calibration.\n";
        return calib;
    }
    
    calib.image_size = images[0].size();
    calib.calibration_method = "Automatic chessboard calibration";
    calib.calibration_date = std::chrono::system_clock::now();
    
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<std::vector<cv::Point3f>> object_points;
    
    std::vector<cv::Point3f> pattern;
    for (int y = 0; y < board_size.height; ++y) {
        for (int x = 0; x < board_size.width; ++x) {
            pattern.push_back(cv::Point3f(x * square_size, y * square_size, 0));
        }
    }
    
    int successful_detections = 0;
    for (const auto& image : images) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, board_size, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
        
        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01));
            
            image_points.push_back(corners);
            object_points.push_back(pattern);
            successful_detections++;
        }
    }
    
    if (successful_detections < Config::calibration.minImages) {
        std::cerr << "Insufficient calibration images (" << successful_detections 
                  << "). Need at least " << Config::calibration.minImages << ".\n";
        return calib;
    }
    
    std::cout << "Using " << successful_detections << " images for calibration...\n";
    
    std::vector<cv::Mat> rvecs, tvecs;
    try {
        calib.reprojection_error = cv::calibrateCamera(
            object_points, image_points, calib.image_size,
            calib.camera_matrix, calib.distortion_coeffs,
            rvecs, tvecs,
            cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_TILTED_MODEL);
    } catch (const cv::Exception& e) {
        rethrowCv(e, "calibrateCamera failed");
    }
    
    if (!calib.validateCameraMatrix() || !calib.validateDistortionCoeffs()) {
        std::cerr << "Calibration produced invalid parameters.\n";
        calib.is_valid = false;
        return calib;
    }
    
    calib.is_valid = true;
    std::cout << "Calibration successful!\n";
    std::cout << "RMS reprojection error: " << calib.reprojection_error << " pixels\n";
    
    double total_error = 0;
    int total_points = 0;
    
    for (size_t i = 0; i < object_points.size(); ++i) {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(object_points[i], rvecs[i], tvecs[i], 
                         calib.camera_matrix, calib.distortion_coeffs, projected);
        
        for (size_t j = 0; j < projected.size(); ++j) {
            double error = cv::norm(image_points[i][j] - projected[j]);
            total_error += error * error;
            total_points++;
        }
    }
    
    double mean_error = std::sqrt(total_error / total_points);
    std::cout << "Mean per-point error: " << mean_error << " pixels\n";
    
    return calib;
}

void saveCalibration(const std::string& filename, const CameraCalibration& calib) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    
    if (!fs.isOpened()) {
        std::cerr << "Cannot open file for writing: " << filename << "\n";
        return;
    }
    
    fs << "camera_matrix" << calib.camera_matrix;
    fs << "distortion_coefficients" << calib.distortion_coeffs;
    fs << "image_width" << calib.image_size.width;
    fs << "image_height" << calib.image_size.height;
    fs << "calibration_method" << calib.calibration_method;
    fs << "reprojection_error" << calib.reprojection_error;
    
    auto time_t = std::chrono::system_clock::to_time_t(calib.calibration_date);
    fs << "calibration_timestamp" << std::ctime(&time_t);
    
    auto [hfov, vfov] = calib.getFieldOfView();
    fs << "horizontal_fov_degrees" << hfov;
    fs << "vertical_fov_degrees" << vfov;
    fs << "is_valid" << calib.is_valid;
    
    fs.release();
    std::cout << "Calibration saved to: " << filename << "\n";
}
