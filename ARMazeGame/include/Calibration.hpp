#pragma once
#include <opencv2/opencv.hpp>
#include <string>

/**
 * @brief Runs live camera calibration with chessboard pattern
 * @param camIndex Camera device index
 * @param yamlPath Output YAML file path
 * @param calibSize_out Output calibration image size
 * @param K_out Output intrinsic matrix
 * @param D_out Output distortion coefficients
 * @return true if calibration successful
 */
bool runLiveCalibration(int camIndex,
                        const std::string& yamlPath,
                        cv::Size& calibSize_out,
                        cv::Mat& K_out, cv::Mat& D_out);

/**
 * @brief Creates default calibration parameters
 */
void createDefaultCalibration(int width, int height, cv::Mat& K, cv::Mat& D);
