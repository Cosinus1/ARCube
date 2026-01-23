#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Draws labeled quadrilateral with corner markers
 */
void drawLabeledQuad(cv::Mat& frame, const std::vector<cv::Point2f>& q);

/**
 * @brief Draws 3D axes overlay
 */
void drawAxes(cv::Mat& frame, const cv::Mat& K, const cv::Mat& D,
              const cv::Mat& rvec, const cv::Mat& tvec, float L = 0.2f);

/**
 * @brief Draws cube wireframe
 */
void drawCubeWireframe(cv::Mat& frame, const std::vector<cv::Point2f>& imgPts);

/**
 * @brief Draws cube faces with depth sorting
 */
void drawCubeFaces(cv::Mat& frame,
                   const std::vector<cv::Point3f>& cubePts3D,
                   const cv::Mat& rvec, const cv::Mat& tvec,
                   const cv::Mat& K, const cv::Mat& D);

/**
 * @brief Draws detection status overlay
 */
void drawStatusOverlay(cv::Mat& frame, bool detected, double confidence, 
                       const std::string& status);
