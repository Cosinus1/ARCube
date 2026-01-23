#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

void drawLabeledQuad(cv::Mat& frame, const std::vector<cv::Point2f>& q);

void drawAxes(cv::Mat& frame, const cv::Mat& K, const cv::Mat& D,
              const cv::Mat& rvec, const cv::Mat& tvec, float L=0.2f);

void drawCubeWireframe(cv::Mat& frame, const std::vector<cv::Point2f>& imgPts);

void drawCubeFaces(cv::Mat& frame,
                   const std::vector<cv::Point3f>& cubePts3D,
                   const cv::Mat& rvec, const cv::Mat& tvec,
                   const cv::Mat& K, const cv::Mat& D);
