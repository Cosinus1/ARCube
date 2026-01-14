/**
 * @file Calibration.hpp
 * @brief Camera calibration management
 * @author Olivier Deruelle
 * @date 2025
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <utility>

/**
 * @brief Camera calibration parameters structure
 */
struct CameraCalibration {
    cv::Mat camera_matrix;
    cv::Mat distortion_coeffs;
    cv::Size image_size;
    double reprojection_error;
    std::string calibration_method;
    std::chrono::system_clock::time_point calibration_date;
    bool is_valid;
    
    CameraCalibration();
    bool validateCameraMatrix() const;
    bool validateDistortionCoeffs() const;
    CameraCalibration scaleToSize(const cv::Size& new_size) const;
    std::pair<double, double> getFieldOfView() const;
    void printSummary() const;
};

// Global camera parameters (for compatibility)
extern cv::Mat cameraMatrix;
extern cv::Mat distCoeffs;
extern CameraCalibration current_calibration;

/**
 * @brief Builds default camera calibration
 */
void buildDefaultCalibration(int width, int height);

/**
 * @brief Loads camera calibration from YAML file
 */
void loadCalibration(const std::string& filename, int width, int height);

/**
 * @brief Performs automatic camera calibration
 */
CameraCalibration performAutoCalibration(const std::vector<cv::Mat>& images,
                                         const cv::Size& board_size,
                                         float square_size);

/**
 * @brief Saves calibration to YAML file
 */
void saveCalibration(const std::string& filename, const CameraCalibration& calib);
