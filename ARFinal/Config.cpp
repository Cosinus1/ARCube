/**
 * @file Config.cpp
 * @brief Implementation of centralized configuration management
 */

#include "Config.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

namespace Config {
    // Define global configuration instances
    Detection detection;
    Pose pose;
    Render render;
    Calibration calibration;
    Runtime runtime;

    bool load(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        runtime.sourceConfigPath = filename;
        
        if (!fs.isOpened()) {
            std::cerr << "Config file not found (" << filename << "), using defaults.\n";
            return false;
        }

        auto get = [&](const char* key, auto& target) {
            if (!fs[key].empty()) fs[key] >> target;
        };

        // Detection parameters
        get("detection_history_size", detection.historySize);
        get("detection_min_confidence", detection.minConfidence);
        get("detection_min_detections", detection.minDetections);
        get("detection_adaptive_roi_expansion", detection.adaptiveRoiExpansion);
        
        if (!fs["detection_scales"].empty()) {
            detection.multiscaleScales.clear();
            cv::FileNode n = fs["detection_scales"];
            for (auto it = n.begin(); it != n.end(); ++it) {
                detection.multiscaleScales.push_back((double)*it);
            }
        }

        // Pose parameters
        get("pose_history_size", pose.poseHistorySize);
        get("pose_max_translation_jump", pose.maxTranslationJump);
        get("pose_max_rotation_jump", pose.maxRotationJump);
        get("pose_min_reprojection_quality", pose.minReprojectionQuality);
        get("pose_min_inliers_ratio", pose.minInliersRatio);
        get("pose_smoothing_alpha", pose.smoothingAlpha);
        get("quad_smoothing_alpha", pose.quadSmoothingAlpha);

        // Render parameters
        get("render_line_thickness_px", render.lineThicknessPx);
        get("render_max_window_w", render.maxWindowW);
        get("render_max_window_h", render.maxWindowH);

        // Calibration parameters
        get("calib_min_images", calibration.minImages);
        get("calib_cols", calibration.chessboardCols);
        get("calib_rows", calibration.chessboardRows);
        get("calib_square_size_mm", calibration.chessboardSquareSize);

        // Runtime parameters
        get("runtime_verbose", runtime.verbose);
        get("runtime_show_pose_metrics", runtime.showPoseMetrics);
        get("runtime_pose_metrics_interval", runtime.poseMetricsInterval);
        get("runtime_detection_stats_interval", runtime.detectionStatsInterval);

        std::cout << "Loaded configuration from: " << filename << "\n";
        return true;
    }

    void printSummary() {
        std::cout << "\n=== CONFIG SUMMARY ===\n";
        std::cout << "Config path: " << runtime.sourceConfigPath << "\n";
        std::cout << "Detection: history=" << detection.historySize
                  << " minConf=" << detection.minConfidence
                  << " minDet=" << detection.minDetections
                  << " roiExpand=" << detection.adaptiveRoiExpansion
                  << " scales=";
        for (auto s : detection.multiscaleScales) std::cout << s << " ";
        std::cout << "\nPose: hist=" << pose.poseHistorySize
                  << " maxTransJump=" << pose.maxTranslationJump
                  << " maxRotJump=" << pose.maxRotationJump
                  << " minReprojQual=" << pose.minReprojectionQuality
                  << " minInliersRatio=" << pose.minInliersRatio
                  << " smoothAlpha=" << pose.smoothingAlpha
                  << " quadAlpha=" << pose.quadSmoothingAlpha << "\n";
        std::cout << "Render: lineThicknessPx=" << render.lineThicknessPx
                  << " maxWindow=" << render.maxWindowW << "x" << render.maxWindowH << "\n";
        std::cout << "Calibration: minImages=" << calibration.minImages
                  << " board=" << calibration.chessboardCols << "x" << calibration.chessboardRows
                  << " squareSize(mm)=" << calibration.chessboardSquareSize << "\n";
        std::cout << "Runtime: verbose=" << (runtime.verbose ? "true" : "false")
                  << " poseMetricsInterval=" << runtime.poseMetricsInterval
                  << " detectionStatsInterval=" << runtime.detectionStatsInterval << "\n";
        std::cout << "=======================\n\n";
    }

} // namespace Config
