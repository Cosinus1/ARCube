#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

bool poseFromQuadHomography(const std::vector<cv::Point2f>& quadTLTRBRBL,
                            const cv::Mat& K, const cv::Mat& D,
                            double side, cv::Mat& rvec, cv::Mat& tvec);

void projectSquareBase(double side, const cv::Mat& K, const cv::Mat& D,
                       const cv::Mat& rvec, const cv::Mat& tvec,
                       std::vector<cv::Point2f>& out2d);
