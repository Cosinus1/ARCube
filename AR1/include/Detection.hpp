#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::Point2f> orderQuadRobust(const std::vector<cv::Point2f>& quadRaw);

bool findLargestQuad(const cv::Mat& gray, std::vector<cv::Point2f>& quadOrdered);
