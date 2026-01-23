#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief Quaternion structure for smooth rotation interpolation
 */
struct Quat { 
    double w, x, y, z; 
    
    Quat() : w(1), x(0), y(0), z(0) {}
    Quat(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}
};

/**
 * @brief Convert rotation matrix to quaternion
 */
Quat matToQuat(const cv::Matx33d& R);

/**
 * @brief Convert quaternion to rotation matrix
 */
cv::Matx33d quatToMat(const Quat& q);

/**
 * @brief Normalize quaternion to unit length
 */
Quat quatNorm(const Quat& a);

/**
 * @brief Spherical linear interpolation between quaternions
 */
Quat quatSlerp(const Quat& qa, const Quat& qb, double t);

/**
 * @brief Convert rotation vector to quaternion
 */
Quat rotvecToQuat(const cv::Mat& rvec);

/**
 * @brief Convert quaternion to rotation vector
 */
cv::Mat quatToRotvec(const Quat& q);
