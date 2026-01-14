/**
 * @file Physics.hpp
 * @brief Physics simulation for ball and maze game
 * @author AR Cube Project
 * @date 2025
 * 
 * @details Simulates ball physics on a tilted plane with maze walls.
 * The ball moves based on gravity projected onto the tilted plane surface.
 */

#pragma once

#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Wall segment for collision detection
 */
struct WallSegment {
    glm::vec2 p1;      // Start point
    glm::vec2 p2;      // End point
    float thickness;   // Wall thickness
    
    WallSegment(float x1, float y1, float x2, float y2, float thick)
        : p1(x1, y1), p2(x2, y2), thickness(thick) {}
};

/**
 * @brief Ball physics state
 */
struct BallState {
    glm::vec2 position;     // Current position on the plane (x, y)
    glm::vec2 velocity;     // Current velocity
    float radius;           // Ball radius
    
    BallState() : position(0.f, 0.f), velocity(0.f, 0.f), radius(10.f) {}
};

/**
 * @brief Physics simulator for ball and maze
 */
class PhysicsSimulator {
public:
    PhysicsSimulator();
    
    /**
     * @brief Initialize the physics simulator
     * @param halfW Half width of the play area
     * @param halfH Half height of the play area
     * @param ballRadius Radius of the ball
     * @param wallThickness Thickness of walls
     */
    void initialize(float halfW, float halfH, float ballRadius, float wallThickness);
    
    /**
     * @brief Reset ball to starting position
     */
    void reset();
    
    /**
     * @brief Update physics simulation
     * @param rvec Rotation vector from pose estimation
     * @param deltaTime Time step in seconds
     */
    void update(const cv::Mat& rvec, float deltaTime);
    
    /**
     * @brief Get current ball position
     */
    glm::vec2 getBallPosition() const { return m_ball.position; }
    
    /**
     * @brief Get current ball velocity
     */
    glm::vec2 getBallVelocity() const { return m_ball.velocity; }
    
    /**
     * @brief Get ball radius
     */
    float getBallRadius() const { return m_ball.radius; }
    
    /**
     * @brief Check if ball reached goal (center of maze)
     */
    bool isAtGoal() const;
    
    /**
     * @brief Get tilt angles (for debugging/visualization)
     */
    glm::vec2 getTilt() const { return m_currentTilt; }
    
private:
    /**
     * @brief Calculate tilt from rotation vector
     */
    glm::vec2 calculateTilt(const cv::Mat& rvec);
    
    /**
     * @brief Calculate gravity acceleration on tilted plane
     */
    glm::vec2 calculateGravityAcceleration(const glm::vec2& tilt);
    
    /**
     * @brief Check and resolve collisions with walls
     */
    void resolveWallCollisions();
    
    /**
     * @brief Check and resolve collision with boundary
     */
    void resolveBoundaryCollisions();
    
    /**
     * @brief Check collision with a single wall segment
     * @return true if collision occurred
     */
    bool checkWallCollision(const WallSegment& wall, glm::vec2& normal, float& penetration);
    
    /**
     * @brief Get closest point on line segment to a point
     */
    glm::vec2 closestPointOnSegment(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b);
    
    // State
    BallState m_ball;
    glm::vec2 m_currentTilt;
    
    // Configuration
    float m_halfW;
    float m_halfH;
    float m_wallThickness;
    float m_gravity;        // Gravity constant
    float m_friction;       // Rolling friction coefficient
    float m_bounciness;     // Coefficient of restitution
    float m_maxVelocity;    // Maximum velocity cap
    
    // Walls
    std::vector<WallSegment> m_mazeWalls;
    std::vector<WallSegment> m_boundaryWalls;
    
    bool m_initialized;
};

/**
 * @brief Create default maze wall segments
 */
std::vector<WallSegment> createDefaultMazeWalls(float halfW, float halfH, float thickness);

/**
 * @brief Create boundary wall segments
 */
std::vector<WallSegment> createBoundaryWalls(float halfW, float halfH, float thickness);
