#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

/**
 * @brief Resolves the data root directory
 */
std::filesystem::path resolveDataRoot(const std::filesystem::path& exePath);

/**
 * @brief Loads camera calibration from YAML file
 */
bool loadCalibration(const std::string& yamlPath, cv::Mat& K, cv::Mat& D, cv::Size& calibSize);

/**
 * @brief Scales intrinsic matrix to new image size
 */
cv::Mat scaleIntrinsics(const cv::Mat& K0, cv::Size from, cv::Size to);

/**
 * @brief Rotation enumeration for frame orientation
 */
enum class Rot { NONE, CW, CCW };

/**
 * @brief Rotates frame in place
 */
void rotateFrameInPlace(cv::Mat& img, Rot r);

/**
 * @brief Determines required rotation based on calibration and frame sizes
 */
Rot decideRotationFromSizes(const cv::Size& calibSize, const cv::Size& frameSize);

/**
 * @brief Lists files with specific extensions in a directory
 */
std::vector<std::string> listFilesInDirectory(const std::filesystem::path& dir, 
                                               const std::vector<std::string>& extensions);

/**
 * @brief Gets list of available webcams
 */
std::vector<int> getAvailableWebcams(int maxCheck = 5);
