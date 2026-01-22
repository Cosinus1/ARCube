#pragma once
#include <opencv2/opencv.hpp>
#include <string>

bool runLiveCalibration(int camIndex,
                        const std::string& yamlPath,
                        cv::Size& calibSize_out,
                        cv::Mat& K_out, cv::Mat& D_out);
