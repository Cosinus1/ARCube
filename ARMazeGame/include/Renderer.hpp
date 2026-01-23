#pragma once
#include <opencv2/opencv.hpp>
#include "Physics.hpp"
#include "Maze.hpp"
#include <vector>

/**
 * @brief Renderer for the AR maze game using OpenCV
 */
class MazeRenderer {
public:
    MazeRenderer();
    
    /**
     * @brief Render the complete maze scene
     * @param frame Output frame to draw on
     * @param K Camera intrinsic matrix
     * @param D Distortion coefficients
     * @param rvec Rotation vector
     * @param tvec Translation vector
     * @param physics Physics simulator with game state
     * @param gameState Current game state
     */
    void render(cv::Mat& frame,
                const cv::Mat& K, const cv::Mat& D,
                const cv::Mat& rvec, const cv::Mat& tvec,
                const PhysicsSimulator& physics,
                const GameState& gameState);
    
    /**
     * @brief Render just the plane outline
     */
    void renderPlaneOutline(cv::Mat& frame,
                            const cv::Mat& K, const cv::Mat& D,
                            const cv::Mat& rvec, const cv::Mat& tvec,
                            float halfW, float halfH);
    
    /**
     * @brief Render walls
     */
    void renderWalls(cv::Mat& frame,
                     const cv::Mat& K, const cv::Mat& D,
                     const cv::Mat& rvec, const cv::Mat& tvec,
                     const std::vector<WallSegment>& walls,
                     float wallHeight,
                     const cv::Scalar& color);
    
    /**
     * @brief Render the ball
     */
    void renderBall(cv::Mat& frame,
                    const cv::Mat& K, const cv::Mat& D,
                    const cv::Mat& rvec, const cv::Mat& tvec,
                    const Vec2& position, float radius);
    
    /**
     * @brief Render the goal/hole
     */
    void renderGoal(cv::Mat& frame,
                    const cv::Mat& K, const cv::Mat& D,
                    const cv::Mat& rvec, const cv::Mat& tvec,
                    const Goal& goal);
    
    /**
     * @brief Render game HUD (score, time, etc.)
     */
    void renderHUD(cv::Mat& frame, const GameState& gameState, 
                   const Vec2& tilt, bool detected);
    
    /**
     * @brief Render win screen overlay
     */
    void renderWinScreen(cv::Mat& frame, const GameState& gameState);

    // Configuration
    void setWallHeight(float height) { wallHeight = height; }
    void setWallColor(const cv::Scalar& color) { wallColor = color; }
    void setBallColor(const cv::Scalar& color) { ballColor = color; }
    void setGoalColor(const cv::Scalar& color) { goalColor = color; }
    void setPlaneColor(const cv::Scalar& color) { planeColor = color; }

private:
    // Helper functions
    std::vector<cv::Point2f> projectPoints3D(const std::vector<cv::Point3f>& points3D,
                                             const cv::Mat& K, const cv::Mat& D,
                                             const cv::Mat& rvec, const cv::Mat& tvec);
    
    void drawFilledPolygon(cv::Mat& frame, const std::vector<cv::Point2f>& points,
                           const cv::Scalar& color, double alpha = 0.7);
    
    void drawWallSegment3D(cv::Mat& frame,
                           const cv::Mat& K, const cv::Mat& D,
                           const cv::Mat& rvec, const cv::Mat& tvec,
                           const WallSegment& wall, float height,
                           const cv::Scalar& color);
    
    // Rendering parameters
    float wallHeight;
    cv::Scalar wallColor;
    cv::Scalar ballColor;
    cv::Scalar goalColor;
    cv::Scalar planeColor;
    cv::Scalar boundaryColor;
};
