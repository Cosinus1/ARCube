#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Estimates pose from quadrilateral using homography decomposition
 * @param quadTLTRBRBL Ordered image points (TL, TR, BR, BL)
 * @param K Camera intrinsic matrix
 * @param D Distortion coefficients
 * @param side Physical size of the marker (in meters or mm)
 * @param rvec Output rotation vector
 * @param tvec Output translation vector
 * @return true if pose estimation successful
 */
bool poseFromQuadHomography(const std::vector<cv::Point2f>& quadTLTRBRBL,
                            const cv::Mat& K, const cv::Mat& D,
                            double side, cv::Mat& rvec, cv::Mat& tvec);

/**
 * @brief Projects square base points to image
 */
void projectSquareBase(double side, const cv::Mat& K, const cv::Mat& D,
                       const cv::Mat& rvec, const cv::Mat& tvec,
                       std::vector<cv::Point2f>& out2d);

/**
 * @brief Computes reprojection error for pose validation
 */
double computeReprojectionError(const std::vector<cv::Point2f>& imagePoints,
                                const std::vector<cv::Point2f>& projectedPoints);
