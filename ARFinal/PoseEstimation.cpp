/**
 * @file PoseEstimation.cpp
 * @brief Implementation of camera pose estimation utilities
 */

#include "PoseEstimation.hpp"
#include "Config.hpp"
#include "Exceptions.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

// ============= Quaternion Math =============
Quat matToQuat(const cv::Matx33d& R) {
    Quat q;
    double tr = R(0, 0) + R(1, 1) + R(2, 2);
    if (tr > 0) {
        double s = std::sqrt(tr + 1.0) * 2.0;
        q.w = 0.25 * s;
        q.x = (R(2, 1) - R(1, 2)) / s;
        q.y = (R(0, 2) - R(2, 0)) / s;
        q.z = (R(1, 0) - R(0, 1)) / s;
    } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
        double s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
        q.w = (R(2, 1) - R(1, 2)) / s;
        q.x = 0.25 * s;
        q.y = (R(0, 1) + R(1, 0)) / s;
        q.z = (R(0, 2) + R(2, 0)) / s;
    } else if (R(1, 1) > R(2, 2)) {
        double s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
        q.w = (R(0, 2) - R(2, 0)) / s;
        q.x = (R(0, 1) + R(1, 0)) / s;
        q.y = 0.25 * s;
        q.z = (R(1, 2) + R(2, 1)) / s;
    } else {
        double s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
        q.w = (R(1, 0) - R(0, 1)) / s;
        q.x = (R(0, 2) + R(2, 0)) / s;
        q.y = (R(1, 2) + R(2, 1)) / s;
        q.z = 0.25 * s;
    }
    return q;
}

cv::Matx33d quatToMat(const Quat& q) {
    double w = q.w, x = q.x, y = q.y, z = q.z;
    return cv::Matx33d(
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w,   2 * x * z + 2 * y * w,
        2 * x * y + 2 * z * w,   1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
        2 * x * z - 2 * y * w,   2 * y * z + 2 * x * w,   1 - 2 * x * x - 2 * y * y
    );
}

Quat quatNorm(const Quat& a) {
    double n = std::sqrt(a.w * a.w + a.x * a.x + a.y * a.y + a.z * a.z);
    return {a.w / n, a.x / n, a.y / n, a.z / n};
}

Quat quatSlerp(const Quat& qa, const Quat& qb, double t) {
    Quat a = quatNorm(qa), b = quatNorm(qb);
    double dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    if (dot < 0) {
        dot = -dot;
        b.w = -b.w;
        b.x = -b.x;
        b.y = -b.y;
        b.z = -b.z;
    }
    const double TH = 0.9995;
    if (dot > TH) {
        Quat r{a.w + t * (b.w - a.w), a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z)};
        return quatNorm(r);
    }
    double th0 = std::acos(dot), th = th0 * t;
    double s0 = std::cos(th) - dot * std::sin(th) / std::sin(th0);
    double s1 = std::sin(th) / std::sin(th0);
    return quatNorm({a.w * s0 + b.w * s1, a.x * s0 + b.x * s1, a.y * s0 + b.y * s1, a.z * s0 + b.z * s1});
}

Quat rotvecToQuat(const cv::Mat& rvec) {
    cv::Matx33d R;
    cv::Rodrigues(rvec, R);
    return matToQuat(R);
}

cv::Mat quatToRotvec(const Quat& q) {
    cv::Matx33d R = quatToMat(q);
    cv::Mat r;
    cv::Rodrigues(R, r);
    return r;
}

// ============= Pose Estimation =============
bool poseFromPlanarPoints(const std::vector<cv::Point3f>& obj3,
                          const std::vector<cv::Point2f>& quadTLTRBRBL,
                          const cv::Mat& K, const cv::Mat& D,
                          cv::Mat& rvec, cv::Mat& tvec)
{
    if (obj3.size() != 4 || quadTLTRBRBL.size() != 4) return false;
    try {
        std::vector<cv::Point2f> obj2(4);
        for (int i = 0; i < 4; ++i) obj2[i] = cv::Point2f(obj3[i].x, obj3[i].y);

        std::vector<cv::Point2f> und;
        cv::undistortPoints(quadTLTRBRBL, und, K, D);

        cv::Mat H = cv::getPerspectiveTransform(obj2, und);

        cv::Mat h1 = H.col(0), h2 = H.col(1), h3 = H.col(2);
        double n1 = cv::norm(h1), n2 = cv::norm(h2);
        double lambda = 1.0 / ((n1 + n2) * 0.5);

        cv::Mat r1 = lambda * h1;
        cv::Mat r2 = lambda * h2;

        r1 /= cv::norm(r1);
        r2 = r2 - (r1.dot(r2)) * r1;
        r2 /= cv::norm(r2);
        cv::Mat r3 = r1.cross(r2);

        cv::Mat R(3, 3, CV_64F);
        r1.copyTo(R.col(0));
        r2.copyTo(R.col(1));
        r3.copyTo(R.col(2));

        cv::Mat t = lambda * h3;

        if (t.at<double>(2) < 0) {
            R = -R;
            t = -t;
        }

        cv::Rodrigues(R, rvec);
        tvec = t.clone();
    } catch (const cv::Exception& e) {
        std::cerr << "poseFromPlanarPoints error: " << e.what() << std::endl;
        return false;
    }
    return true;
}

// ============= RobustPoseEstimator =============
RobustPoseEstimator::RobustPoseEstimator() 
    : last_confidence(0.0), is_initialized(false), consecutive_good_poses(0) {}

double RobustPoseEstimator::estimateRobustPose(const std::vector<cv::Point3f>& object_points,
                                              const std::vector<cv::Point2f>& image_points,
                                              const cv::Mat& camera_matrix,
                                              const cv::Mat& dist_coeffs,
                                              cv::Mat& rvec,
                                              cv::Mat& tvec)
{
    if (object_points.size() != 4 || image_points.size() != 4) {
        return -1.0;
    }
    
    std::vector<PoseCandidate> candidates;
    
    // Method 1: Homography-based
    cv::Mat rvec1, tvec1;
    if (poseFromPlanarPoints(object_points, image_points, camera_matrix, dist_coeffs, rvec1, tvec1)) {
        double error1 = computeReprojectionError(object_points, image_points, rvec1, tvec1, camera_matrix, dist_coeffs);
        candidates.push_back({rvec1.clone(), tvec1.clone(), error1, "Homography"});
    }
    
    // Method 2: IPPE
    cv::Mat rvec2, tvec2;
    try {
        if (cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec2, tvec2, false, cv::SOLVEPNP_IPPE)) {
            double error2 = computeReprojectionError(object_points, image_points, rvec2, tvec2, camera_matrix, dist_coeffs);
            candidates.push_back({rvec2.clone(), tvec2.clone(), error2, "IPPE"});
        }
    } catch (const cv::Exception& e) {
        rethrowCv(e, "solvePnP IPPE");
    }
    
    // Method 3: IPPE Square
    cv::Mat rvec3, tvec3;
    try {
        if (cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec3, tvec3, false, cv::SOLVEPNP_IPPE_SQUARE)) {
            double error3 = computeReprojectionError(object_points, image_points, rvec3, tvec3, camera_matrix, dist_coeffs);
            candidates.push_back({rvec3.clone(), tvec3.clone(), error3, "IPPE_SQUARE"});
        }
    } catch (const cv::Exception& e) {
        rethrowCv(e, "solvePnP IPPE_SQUARE");
    }
    
    // Method 4: RANSAC
    cv::Mat rvec4, tvec4;
    std::vector<int> inliers;
    try {
        if (cv::solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs, rvec4, tvec4, 
                              false, 100, 4.0, 0.99, inliers)) {
            double inlier_ratio = (double)inliers.size() / object_points.size() * 100.0;
            if (inlier_ratio >= Config::pose.minInliersRatio) {
                double error4 = computeReprojectionError(object_points, image_points, rvec4, tvec4, camera_matrix, dist_coeffs);
                candidates.push_back({rvec4.clone(), tvec4.clone(), error4, "RANSAC"});
            }
        }
    } catch (const cv::Exception& e) {
        rethrowCv(e, "solvePnPRansac");
    }
    
    // Method 5: Iterative refinement
    if (is_initialized && !last_valid_rvec.empty()) {
        cv::Mat rvec5 = last_valid_rvec.clone();
        cv::Mat tvec5 = last_valid_tvec.clone();
        if (cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec5, tvec5, true, cv::SOLVEPNP_ITERATIVE)) {
            double error5 = computeReprojectionError(object_points, image_points, rvec5, tvec5, camera_matrix, dist_coeffs);
            candidates.push_back({rvec5.clone(), tvec5.clone(), error5, "Iterative"});
        }
    }
    
    if (candidates.empty()) {
        return -1.0;
    }
    
    std::sort(candidates.begin(), candidates.end(), 
              [](const PoseCandidate& a, const PoseCandidate& b) {
                  return a.reprojection_error < b.reprojection_error;
              });
    
    PoseCandidate best_candidate = candidates[0];
    
    if (best_candidate.reprojection_error > Config::pose.minReprojectionQuality) return -1.0;
    
    double temporal_confidence = validateTemporalCoherence(best_candidate.rvec, best_candidate.tvec);
    if (temporal_confidence < 0.3 && is_initialized) {
        if (predictPoseFromHistory(rvec, tvec)) {
            double predicted_error = computeReprojectionError(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs);
            if (predicted_error < best_candidate.reprojection_error * 1.5) {
                updateHistory(rvec, tvec, 0.5, predicted_error);
                return 0.5;
            }
        }
    }
    
    cv::Mat refined_rvec, refined_tvec;
    double refinement_confidence = refinePoseWithSubPixel(object_points, image_points, 
                                                         best_candidate.rvec, best_candidate.tvec,
                                                         camera_matrix, dist_coeffs,
                                                         refined_rvec, refined_tvec);
    
    if (refinement_confidence > 0.6) {
        best_candidate.rvec = refined_rvec;
        best_candidate.tvec = refined_tvec;
        best_candidate.reprojection_error = computeReprojectionError(object_points, image_points, 
                                                                    refined_rvec, refined_tvec, 
                                                                    camera_matrix, dist_coeffs);
    }
    
    double geometric_confidence = calculateGeometricConfidence(best_candidate.rvec, best_candidate.tvec);
    double reprojection_confidence = std::max(0.0, 1.0 - best_candidate.reprojection_error / Config::pose.minReprojectionQuality);
    double final_confidence = (temporal_confidence + geometric_confidence + reprojection_confidence) / 3.0;
    
    if (final_confidence > 0.4) {
        rvec = best_candidate.rvec.clone();
        tvec = best_candidate.tvec.clone();
        updateHistory(rvec, tvec, final_confidence, best_candidate.reprojection_error);
        
        if (!is_initialized && consecutive_good_poses >= 3) {
            is_initialized = true;
        }
        
        if (final_confidence > 0.7) {
            consecutive_good_poses++;
        } else {
            consecutive_good_poses = std::max(0, consecutive_good_poses - 1);
        }
        
        return final_confidence;
    }
    
    return -1.0;
}

RobustPoseEstimator::PoseQualityMetrics RobustPoseEstimator::getQualityMetrics() const {
    PoseQualityMetrics metrics;
    
    if (confidence_history.empty()) {
        metrics.average_confidence = 0.0;
        metrics.average_reprojection_error = 999.0;
        metrics.consecutive_good_poses = 0;
        metrics.is_stable = false;
        metrics.status = "No pose history";
        return metrics;
    }
    
    double conf_sum = 0.0, error_sum = 0.0;
    for (size_t i = 0; i < confidence_history.size(); ++i) {
        conf_sum += confidence_history[i];
        error_sum += reprojection_errors[i];
    }
    
    metrics.average_confidence = conf_sum / confidence_history.size();
    metrics.average_reprojection_error = error_sum / reprojection_errors.size();
    metrics.consecutive_good_poses = consecutive_good_poses;
    metrics.is_stable = is_initialized && metrics.average_confidence > 0.7;
    
    if (metrics.is_stable) {
        metrics.status = "EXCELLENT";
    } else if (metrics.average_confidence > 0.5) {
        metrics.status = "GOOD";
    } else if (metrics.average_confidence > 0.3) {
        metrics.status = "FAIR";
    } else {
        metrics.status = "POOR";
    }
    
    return metrics;
}

double RobustPoseEstimator::computeReprojectionError(const std::vector<cv::Point3f>& object_points,
                                                    const std::vector<cv::Point2f>& image_points,
                                                    const cv::Mat& rvec, const cv::Mat& tvec,
                                                    const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs)
{
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected_points);
    
    double total_error = 0.0;
    for (size_t i = 0; i < image_points.size(); ++i) {
        cv::Point2f diff = image_points[i] - projected_points[i];
        total_error += std::sqrt(diff.x * diff.x + diff.y * diff.y);
    }
    
    return total_error / image_points.size();
}

double RobustPoseEstimator::validateTemporalCoherence(const cv::Mat& rvec, const cv::Mat& tvec) {
    if (!is_initialized || rvec_history.empty()) return 1.0;
    
    cv::Mat last_rvec = rvec_history.back();
    cv::Mat last_tvec = tvec_history.back();
    
    cv::Mat translation_diff = tvec - last_tvec;
    double translation_jump = cv::norm(translation_diff);
    double translation_score = std::max(0.0, 1.0 - translation_jump / Config::pose.maxTranslationJump);
    
    cv::Mat rotation_diff = rvec - last_rvec;
    double rotation_jump = cv::norm(rotation_diff);
    double rotation_score = std::max(0.0, 1.0 - rotation_jump / Config::pose.maxRotationJump);
    
    double velocity_score = 1.0;
    if (rvec_history.size() >= 2) {
        cv::Mat prev_translation_diff = last_tvec - tvec_history[tvec_history.size() - 2];
        cv::Mat prev_rotation_diff = last_rvec - rvec_history[rvec_history.size() - 2];
        
        double translation_acceleration = cv::norm(translation_diff - prev_translation_diff);
        double rotation_acceleration = cv::norm(rotation_diff - prev_rotation_diff);
        
        velocity_score = std::max(0.0, 1.0 - (translation_acceleration + rotation_acceleration) / 
                                  (Config::pose.maxTranslationJump + Config::pose.maxRotationJump));
    }
    
    return (translation_score + rotation_score + velocity_score) / 3.0;
}

bool RobustPoseEstimator::predictPoseFromHistory(cv::Mat& rvec, cv::Mat& tvec) {
    if (rvec_history.size() < 2) return false;
    
    cv::Mat last_rvec = rvec_history.back();
    cv::Mat last_tvec = tvec_history.back();
    cv::Mat prev_rvec = rvec_history[rvec_history.size() - 2];
    cv::Mat prev_tvec = tvec_history[tvec_history.size() - 2];
    
    cv::Mat translation_velocity = last_tvec - prev_tvec;
    cv::Mat rotation_velocity = last_rvec - prev_rvec;
    
    double damping_factor = 0.7;
    rvec = last_rvec + rotation_velocity * damping_factor;
    tvec = last_tvec + translation_velocity * damping_factor;
    
    return true;
}

double RobustPoseEstimator::refinePoseWithSubPixel(const std::vector<cv::Point3f>& object_points,
                                                  const std::vector<cv::Point2f>& image_points,
                                                  const cv::Mat& initial_rvec, const cv::Mat& initial_tvec,
                                                  const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                                                  cv::Mat& refined_rvec, cv::Mat& refined_tvec)
{
    refined_rvec = initial_rvec.clone();
    refined_tvec = initial_tvec.clone();
    
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 20, 1e-6);
    
    try {
        cv::solvePnPRefineLM(object_points, image_points, camera_matrix, dist_coeffs, 
                            refined_rvec, refined_tvec, criteria);
        
        double initial_error = computeReprojectionError(object_points, image_points, initial_rvec, initial_tvec, camera_matrix, dist_coeffs);
        double refined_error = computeReprojectionError(object_points, image_points, refined_rvec, refined_tvec, camera_matrix, dist_coeffs);
        
        if (refined_error < initial_error) {
            return std::max(0.6, 1.0 - refined_error / Config::pose.minReprojectionQuality);
        } else {
            refined_rvec = initial_rvec.clone();
            refined_tvec = initial_tvec.clone();
            return 0.5;
        }
    } catch (const cv::Exception& e) {
        refined_rvec = initial_rvec.clone();
        refined_tvec = initial_tvec.clone();
        return 0.3;
    }
}

double RobustPoseEstimator::calculateGeometricConfidence(const cv::Mat& rvec, const cv::Mat& tvec) {
    double distance = cv::norm(tvec);
    double distance_score = 1.0;
    if (distance < 200.0 || distance > 2000.0) {
        distance_score = std::max(0.0, 1.0 - std::abs(distance - 600.0) / 1000.0);
    }
    
    double rotation_magnitude = cv::norm(rvec);
    double rotation_score = std::max(0.0, 1.0 - rotation_magnitude / (CV_PI * 0.75));
    
    double stability_score = 1.0;
    if (!confidence_history.empty()) {
        double recent_confidence = confidence_history.back();
        stability_score = (recent_confidence + 1.0) / 2.0;
    }
    
    return (distance_score + rotation_score + stability_score) / 3.0;
}

void RobustPoseEstimator::updateHistory(const cv::Mat& rvec, const cv::Mat& tvec, double confidence, double reprojection_error) {
    cv::Mat smoothed_rvec = rvec.clone();
    cv::Mat smoothed_tvec = tvec.clone();
    
    if (!rvec_history.empty() && confidence > 0.5) {
        double smoothing_factor = std::min(0.3, 1.0 - confidence);
        smoothed_rvec = rvec * (1.0 - smoothing_factor) + rvec_history.back() * smoothing_factor;
        smoothed_tvec = tvec * (1.0 - smoothing_factor) + tvec_history.back() * smoothing_factor;
    }
    
    rvec_history.push_back(smoothed_rvec.clone());
    tvec_history.push_back(smoothed_tvec.clone());
    confidence_history.push_back(confidence);
    reprojection_errors.push_back(reprojection_error);
    
    if (rvec_history.size() > Config::pose.poseHistorySize) {
        rvec_history.erase(rvec_history.begin());
        tvec_history.erase(tvec_history.begin());
        confidence_history.erase(confidence_history.begin());
        reprojection_errors.erase(reprojection_errors.begin());
    }
    
    last_valid_rvec = smoothed_rvec.clone();
    last_valid_tvec = smoothed_tvec.clone();
    last_confidence = confidence;
}
