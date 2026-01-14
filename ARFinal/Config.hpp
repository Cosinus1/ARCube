/**
 * @file Config.hpp
 * @brief Centralized configuration management for AR application
 * @author Olivier Deruelle
 * @date 2025
 */

#pragma once

#include <string>
#include <vector>

namespace Config {
    /**
     * @brief Detection-related configuration parameters
     */
    struct Detection {
        int historySize = 10;
        double minConfidence = 0.6;
        int minDetections = 3;
        double adaptiveRoiExpansion = 1.8;
        std::vector<double> multiscaleScales{1.0, 0.8, 0.6, 1.25};
    };

    /**
     * @brief Pose estimation configuration parameters
     */
    struct Pose {
        int poseHistorySize = 5;
        double maxTranslationJump = 200.0;
        double maxRotationJump = 0.5;
        double minReprojectionQuality = 15.0;
        int minInliersRatio = 75;
        double smoothingAlpha = 0.45;
        double quadSmoothingAlpha = 0.6;
    };

    /**
     * @brief Rendering configuration parameters
     */
    struct Render {
        float lineThicknessPx = 3.0f;
        int maxWindowW = 1280;
        int maxWindowH = 960;
    };

    /**
     * @brief Calibration configuration parameters
     */
    struct Calibration {
        int minImages = 5;
        int chessboardCols = 9;
        int chessboardRows = 6;
        float chessboardSquareSize = 25.0f;
    };

    /**
     * @brief Runtime behavior configuration
     */
    struct Runtime {
        bool verbose = true;
        bool showPoseMetrics = true;
        int poseMetricsInterval = 60;
        int detectionStatsInterval = 30;
        std::string sourceConfigPath;
    };

    // Global configuration instances
    extern Detection detection;
    extern Pose pose;
    extern Render render;
    extern Calibration calibration;
    extern Runtime runtime;

    /**
     * @brief Load configuration from YAML file
     * @param filename Path to configuration file
     * @return true if loaded successfully
     */
    bool load(const std::string& filename);

    /**
     * @brief Print configuration summary to console
     */
    void printSummary();

} // namespace Config
