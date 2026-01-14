/**
 * @file Detection.hpp
 * @brief A4 sheet detection algorithms and utilities
 * @author Olivier Deruelle
 * @date 2025
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "Config.hpp"

/**
 * @brief Adaptive image processing parameter snapshot
 */
struct AdaptiveParamSnapshot {
    int blurK = 5;
    double cannyLow = 50;
    double cannyHigh = 150;
    int morphK = 5;
    double meanGray = 128;
    double stdGray = 40;
    double confidenceBoost = 0.0;
};

/**
 * @brief Gets last adaptive parameters used
 */
const AdaptiveParamSnapshot& getLastAdaptiveParams();

/**
 * @brief Orders quadrilateral points in consistent sequence
 * @param pts Unordered quadrilateral corners
 * @return Ordered points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
 */
std::vector<cv::Point2f> orderQuadPoints(const std::vector<cv::Point>& pts);

/**
 * @brief Robust quad ordering from Point2f
 */
std::vector<cv::Point2f> orderQuadRobust(const std::vector<cv::Point2f>& quadRaw);

/**
 * @brief Strict A4 sheet detection with aspect ratio validation
 * @param frame Input color image
 * @param corners Output detected corners
 * @return true if valid A4 sheet found
 */
bool detectA4CornersStrict(const cv::Mat& frame, std::vector<cv::Point2f>& corners);

/**
 * @brief Adaptive A4 detection (dynamic pipeline)
 * @param frame Input color image
 * @param corners Output detected corners
 * @param outConfidence Output confidence score
 * @return true if valid A4 sheet found
 */
bool detectA4CornersAdaptive(const cv::Mat& frame,
                             std::vector<cv::Point2f>& corners,
                             double& outConfidence);

/**
 * @brief Temporal validation structure for marker tracking
 */
struct TemporalValidator {
    std::vector<std::vector<cv::Point2f>> detection_history;
    std::vector<double> confidence_history;
    std::vector<cv::Point2f> predicted_corners;
    bool is_initialized;
    int consecutive_detections;
    int frame_count;
    
    TemporalValidator();
    void addDetection(const std::vector<cv::Point2f>& corners, double confidence);
    void updatePrediction();
    double getCurrentConfidence() const;
    bool isValid() const;
    cv::Rect getAdaptiveROI(const cv::Size& img_size, double expansion_factor = 1.5) const;
};

/**
 * @brief Multi-scale detector for robust marker detection
 */
class MultiScaleDetector {
private:
    std::vector<double> scales;
    
public:
    MultiScaleDetector();
    std::pair<std::vector<cv::Point2f>, double> detectMultiScale(const cv::Mat& image, const cv::Rect& roi);
};

/**
 * @brief Robust A4 detection combining multiple techniques
 */
bool detectA4CornersRobust(const cv::Mat& frame,
                           std::vector<cv::Point2f>& corners,
                           TemporalValidator& validator,
                           MultiScaleDetector& detector);
