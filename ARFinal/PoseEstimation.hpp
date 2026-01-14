/**
 * @file PoseEstimation.hpp
 * @brief Camera pose estimation utilities
 * @author Olivier Deruelle
 * @date 2025
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * @brief Quaternion structure for rotation representation
 */
struct Quat {
    double w, x, y, z;
};

/**
 * @brief Converts rotation matrix to quaternion
 */
Quat matToQuat(const cv::Matx33d& R);

/**
 * @brief Converts quaternion to rotation matrix
 */
cv::Matx33d quatToMat(const Quat& q);

/**
 * @brief Normalizes quaternion to unit length
 */
Quat quatNorm(const Quat& a);

/**
 * @brief Spherical linear interpolation between quaternions
 */
Quat quatSlerp(const Quat& qa, const Quat& qb, double t);

/**
 * @brief Converts rotation vector to quaternion
 */
Quat rotvecToQuat(const cv::Mat& rvec);

/**
 * @brief Converts quaternion to rotation vector
 */
cv::Mat quatToRotvec(const Quat& q);

/**
 * @brief Estimates camera pose from planar object points
 */
bool poseFromPlanarPoints(const std::vector<cv::Point3f>& obj3,
                          const std::vector<cv::Point2f>& quadTLTRBRBL,
                          const cv::Mat& K, const cv::Mat& D,
                          cv::Mat& rvec, cv::Mat& tvec);

/**
 * @brief Advanced pose estimation structure with comprehensive validation
 */
struct RobustPoseEstimator {
    struct PoseQualityMetrics {
        double average_confidence;
        double average_reprojection_error;
        int consecutive_good_poses;
        bool is_stable;
        std::string status;
    };
    
    std::vector<cv::Mat> rvec_history;
    std::vector<cv::Mat> tvec_history;
    std::vector<double> confidence_history;
    std::vector<double> reprojection_errors;
    
    cv::Mat last_valid_rvec, last_valid_tvec;
    double last_confidence;
    bool is_initialized;
    int consecutive_good_poses;
    
    RobustPoseEstimator();
    
    double estimateRobustPose(const std::vector<cv::Point3f>& object_points,
                             const std::vector<cv::Point2f>& image_points,
                             const cv::Mat& camera_matrix,
                             const cv::Mat& dist_coeffs,
                             cv::Mat& rvec,
                             cv::Mat& tvec);
    
    PoseQualityMetrics getQualityMetrics() const;
    
private:
    struct PoseCandidate {
        cv::Mat rvec, tvec;
        double reprojection_error;
        std::string method;
    };
    
    double computeReprojectionError(const std::vector<cv::Point3f>& object_points,
                                   const std::vector<cv::Point2f>& image_points,
                                   const cv::Mat& rvec, const cv::Mat& tvec,
                                   const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);
    
    double validateTemporalCoherence(const cv::Mat& rvec, const cv::Mat& tvec);
    bool predictPoseFromHistory(cv::Mat& rvec, cv::Mat& tvec);
    double refinePoseWithSubPixel(const std::vector<cv::Point3f>& object_points,
                                 const std::vector<cv::Point2f>& image_points,
                                 const cv::Mat& initial_rvec, const cv::Mat& initial_tvec,
                                 const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                                 cv::Mat& refined_rvec, cv::Mat& refined_tvec);
    double calculateGeometricConfidence(const cv::Mat& rvec, const cv::Mat& tvec);
    void updateHistory(const cv::Mat& rvec, const cv::Mat& tvec, double confidence, double reprojection_error);
};
