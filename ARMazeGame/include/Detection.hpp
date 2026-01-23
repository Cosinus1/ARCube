#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Orders quadrilateral points robustly (TL, TR, BR, BL order)
 * @param quadRaw Unordered 4 points
 * @return Ordered points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
 */
std::vector<cv::Point2f> orderQuadRobust(const std::vector<cv::Point2f>& quadRaw);

/**
 * @brief Finds the largest quadrilateral in a grayscale image
 * @param gray Input grayscale image
 * @param quadOrdered Output ordered corners
 * @return true if found
 */
bool findLargestQuad(const cv::Mat& gray, std::vector<cv::Point2f>& quadOrdered);

/**
 * @brief Detects A4-like sheet with adaptive parameters
 * @param frame Input color frame
 * @param corners Output detected corners
 * @param confidence Output detection confidence (0.0 - 1.0)
 * @return true if detection successful
 */
bool detectSheetAdaptive(const cv::Mat& frame, 
                         std::vector<cv::Point2f>& corners,
                         double& confidence);
