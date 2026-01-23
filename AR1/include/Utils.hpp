#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

std::filesystem::path resolveDataRoot(const std::filesystem::path& exePath);

bool loadCalibration(const std::string& yamlPath, cv::Mat& K, cv::Mat& D, cv::Size& calibSize);

cv::Mat scaleIntrinsics(const cv::Mat& K0, cv::Size from, cv::Size to);

enum class Rot { NONE, CW, CCW };

void rotateFrameInPlace(cv::Mat& img, Rot r);

Rot decideRotationFromSizes(const cv::Size& calibSize, const cv::Size& frameSize);
