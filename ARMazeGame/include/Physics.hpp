#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 2D Vector for physics calculations
 */
struct Vec2 {
    float x, y;
    
    Vec2() : x(0), y(0) {}
    Vec2(float x_, float y_) : x(x_), y(y_) {}
    
    Vec2 operator+(const Vec2& o) const { return Vec2(x + o.x, y + o.y); }
    Vec2 operator-(const Vec2& o) const { return Vec2(x - o.x, y - o.y); }
    Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator*=(float s) { x *= s; y *= s; return *this; }
    
    float length() const { return std::sqrt(x * x + y * y); }
    float lengthSq() const { return x * x + y * y; }
    Vec2 normalized() const { 
        float len = length(); 
        return len > 0.0001f ? Vec2(x / len, y / len) : Vec2(0, 0); 
    }
    float dot(const Vec2& o) const { return x * o.x + y * o.y; }
};

/**
 * @brief Wall segment for collision detection
 */
struct WallSegment {
    Vec2 p1, p2;        // Start and end points
    float thickness;    // Wall thickness
    
    WallSegment() : thickness(5.0f) {}
    WallSegment(float x1, float y1, float x2, float y2, float thick = 5.0f)
        : p1(x1, y1), p2(x2, y2), thickness(thick) {}
    
    // Get wall normal (perpendicular to wall direction)
    Vec2 getNormal() const {
        Vec2 dir = p2 - p1;
        return Vec2(-dir.y, dir.x).normalized();
    }
};

/**
 * @brief Ball state for physics simulation
 */
struct Ball {
    Vec2 position;      // Current position on the plane
    Vec2 velocity;      // Current velocity
    float radius;       // Ball radius
    
    Ball() : radius(8.0f) {}
};

/**
 * @brief Goal/hole in the maze
 */
struct Goal {
    Vec2 position;      // Center position
    float radius;       // Hole radius
    
    Goal() : radius(15.0f) {}
    Goal(float x, float y, float r) : position(x, y), radius(r) {}
};

/**
 * @brief Physics simulator for the ball and maze game
 */
class PhysicsSimulator {
public:
    PhysicsSimulator();
    
    /**
     * @brief Initialize the physics world
     * @param halfW Half width of play area
     * @param halfH Half height of play area
     * @param ballRadius Ball radius
     */
    void initialize(float halfW, float halfH, float ballRadius);
    
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
     * @brief Add a wall segment to the maze
     */
    void addWall(const WallSegment& wall);
    
    /**
     * @brief Set the goal position
     */
    void setGoal(const Goal& goal);
    
    /**
     * @brief Clear all walls
     */
    void clearWalls();
    
    // Getters
    Vec2 getBallPosition() const { return ball.position; }
    Vec2 getBallVelocity() const { return ball.velocity; }
    float getBallRadius() const { return ball.radius; }
    Vec2 getTilt() const { return currentTilt; }
    Goal getGoal() const { return goal; }
    const std::vector<WallSegment>& getWalls() const { return walls; }
    const std::vector<WallSegment>& getBoundaryWalls() const { return boundaryWalls; }
    
    /**
     * @brief Check if ball reached the goal
     */
    bool isAtGoal() const;
    
    /**
     * @brief Check if ball fell into a hole (game over condition)
     */
    bool isBallInHole() const;

private:
    // Physics calculations
    Vec2 calculateTilt(const cv::Mat& rvec);
    Vec2 calculateGravityAcceleration(const Vec2& tilt);
    void resolveWallCollisions();
    void resolveBoundaryCollisions();
    bool checkWallCollision(const WallSegment& wall, Vec2& normal, float& penetration);
    Vec2 closestPointOnSegment(const Vec2& p, const Vec2& a, const Vec2& b);
    
    // State
    Ball ball;
    Vec2 currentTilt;
    Goal goal;
    
    // World configuration
    float halfW, halfH;
    float gravity;
    float friction;
    float bounciness;
    float maxVelocity;
    
    // Walls
    std::vector<WallSegment> walls;         // Maze walls
    std::vector<WallSegment> boundaryWalls; // Outer boundary
    
    bool initialized;
};
