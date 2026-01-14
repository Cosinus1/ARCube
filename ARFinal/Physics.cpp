/**
 * @file Physics.cpp
 * @brief Implementation of physics simulation for ball and maze game
 * @author AR Cube Project
 * @date 2025
 */

#include "Physics.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <algorithm>

// ============= PhysicsSimulator Implementation =============

PhysicsSimulator::PhysicsSimulator()
    : m_currentTilt(0.f, 0.f)
    , m_halfW(105.f)
    , m_halfH(148.5f)
    , m_wallThickness(8.f)
    , m_gravity(980.f)        // mm/s² (scaled for mm units)
    , m_friction(0.98f)       // Rolling friction (velocity retention per frame)
    , m_bounciness(0.5f)      // Bounce coefficient
    , m_maxVelocity(500.f)    // Maximum velocity cap in mm/s
    , m_initialized(false)
{
}

void PhysicsSimulator::initialize(float halfW, float halfH, float ballRadius, float wallThickness) {
    m_halfW = halfW;
    m_halfH = halfH;
    m_wallThickness = wallThickness;
    m_ball.radius = ballRadius;
    
    // Create maze and boundary walls
    m_mazeWalls = createDefaultMazeWalls(halfW, halfH, wallThickness);
    m_boundaryWalls = createBoundaryWalls(halfW, halfH, wallThickness);
    
    // Start ball at top-left corner (safe starting position)
    m_ball.position = glm::vec2(-halfW * 0.7f, halfH * 0.7f);
    m_ball.velocity = glm::vec2(0.f, 0.f);
    
    m_initialized = true;
}

void PhysicsSimulator::reset() {
    m_ball.position = glm::vec2(-m_halfW * 0.7f, m_halfH * 0.7f);
    m_ball.velocity = glm::vec2(0.f, 0.f);
    m_currentTilt = glm::vec2(0.f, 0.f);
}

glm::vec2 PhysicsSimulator::calculateTilt(const cv::Mat& rvec) {
    // Convert rotation vector to rotation matrix
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    
    // The normal of the plane in camera coordinates is the Z column of R
    // For a flat horizontal plane, the normal would be (0, 0, 1) in world coords
    // After rotation, it becomes R * (0, 0, 1) = third column of R
    
    double nx = R.at<double>(0, 2);
    double ny = R.at<double>(1, 2);
    double nz = R.at<double>(2, 2);
    
    // Calculate tilt angles
    // tilt.x = rotation around Y axis (affects ball movement in X)
    // tilt.y = rotation around X axis (affects ball movement in Y)
    float tiltX = (float)std::atan2(nx, nz);
    float tiltY = (float)std::atan2(-ny, nz);
    
    // Clamp tilt to reasonable values (max ~45 degrees)
    const float maxTilt = glm::radians(45.f);
    tiltX = glm::clamp(tiltX, -maxTilt, maxTilt);
    tiltY = glm::clamp(tiltY, -maxTilt, maxTilt);
    
    return glm::vec2(tiltX, tiltY);
}

glm::vec2 PhysicsSimulator::calculateGravityAcceleration(const glm::vec2& tilt) {
    // Project gravity onto the tilted plane
    // acceleration = g * sin(tilt_angle) for each axis
    float ax = m_gravity * std::sin(tilt.x);
    float ay = m_gravity * std::sin(tilt.y);
    
    return glm::vec2(ax, ay);
}

void PhysicsSimulator::update(const cv::Mat& rvec, float deltaTime) {
    if (!m_initialized || rvec.empty()) return;
    
    // Clamp deltaTime to prevent instability
    deltaTime = std::min(deltaTime, 0.05f);
    
    // Calculate tilt from pose with smoothing
    glm::vec2 newTilt = calculateTilt(rvec);
    
    // Smooth tilt changes to avoid jitter
    const float tiltSmoothing = 0.3f;
    m_currentTilt = m_currentTilt * (1.f - tiltSmoothing) + newTilt * tiltSmoothing;
    
    // Calculate gravity acceleration based on tilt
    glm::vec2 acceleration = calculateGravityAcceleration(m_currentTilt);
    
    // Semi-implicit Euler integration
    // Update velocity first
    m_ball.velocity += acceleration * deltaTime;
    
    // Apply friction (rolling resistance)
    m_ball.velocity *= m_friction;
    
    // Cap velocity
    float speed = glm::length(m_ball.velocity);
    if (speed > m_maxVelocity) {
        m_ball.velocity = (m_ball.velocity / speed) * m_maxVelocity;
    }
    
    // Update position
    m_ball.position += m_ball.velocity * deltaTime;
    
    // Resolve collisions
    resolveWallCollisions();
    resolveBoundaryCollisions();
}

glm::vec2 PhysicsSimulator::closestPointOnSegment(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b) {
    glm::vec2 ab = b - a;
    float t = glm::dot(p - a, ab) / glm::dot(ab, ab);
    t = glm::clamp(t, 0.f, 1.f);
    return a + t * ab;
}

bool PhysicsSimulator::checkWallCollision(const WallSegment& wall, glm::vec2& normal, float& penetration) {
    // Find closest point on wall segment to ball center
    glm::vec2 closest = closestPointOnSegment(m_ball.position, wall.p1, wall.p2);
    
    // Calculate distance from ball center to closest point
    glm::vec2 diff = m_ball.position - closest;
    float dist = glm::length(diff);
    
    // Collision threshold is ball radius + half wall thickness
    float threshold = m_ball.radius + wall.thickness * 0.5f;
    
    if (dist < threshold && dist > 0.001f) {
        normal = diff / dist;  // Normalize
        penetration = threshold - dist;
        return true;
    }
    
    return false;
}

void PhysicsSimulator::resolveWallCollisions() {
    const int maxIterations = 5;  // Prevent infinite loops
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        bool collisionFound = false;
        
        for (const auto& wall : m_mazeWalls) {
            glm::vec2 normal;
            float penetration;
            
            if (checkWallCollision(wall, normal, penetration)) {
                collisionFound = true;
                
                // Push ball out of wall
                m_ball.position += normal * penetration;
                
                // Reflect velocity with bounce
                float normalVelocity = glm::dot(m_ball.velocity, normal);
                if (normalVelocity < 0) {
                    // Reflect and apply bounciness
                    m_ball.velocity -= (1.f + m_bounciness) * normalVelocity * normal;
                    
                    // Apply additional friction on collision
                    m_ball.velocity *= 0.9f;
                }
            }
        }
        
        if (!collisionFound) break;
    }
}

void PhysicsSimulator::resolveBoundaryCollisions() {
    // Simple AABB collision with boundaries
    float effectiveHalfW = m_halfW - m_ball.radius - m_wallThickness;
    float effectiveHalfH = m_halfH - m_ball.radius - m_wallThickness;
    
    // Left boundary
    if (m_ball.position.x < -effectiveHalfW) {
        m_ball.position.x = -effectiveHalfW;
        if (m_ball.velocity.x < 0) {
            m_ball.velocity.x = -m_ball.velocity.x * m_bounciness;
        }
    }
    
    // Right boundary
    if (m_ball.position.x > effectiveHalfW) {
        m_ball.position.x = effectiveHalfW;
        if (m_ball.velocity.x > 0) {
            m_ball.velocity.x = -m_ball.velocity.x * m_bounciness;
        }
    }
    
    // Bottom boundary
    if (m_ball.position.y < -effectiveHalfH) {
        m_ball.position.y = -effectiveHalfH;
        if (m_ball.velocity.y < 0) {
            m_ball.velocity.y = -m_ball.velocity.y * m_bounciness;
        }
    }
    
    // Top boundary
    if (m_ball.position.y > effectiveHalfH) {
        m_ball.position.y = effectiveHalfH;
        if (m_ball.velocity.y > 0) {
            m_ball.velocity.y = -m_ball.velocity.y * m_bounciness;
        }
    }
}

bool PhysicsSimulator::isAtGoal() const {
    // Goal is at center of the maze
    float distToCenter = glm::length(m_ball.position);
    return distToCenter < m_ball.radius * 2.f;
}

// ============= Wall Creation Functions =============

std::vector<WallSegment> createDefaultMazeWalls(float halfW, float halfH, float thickness) {
    std::vector<WallSegment> walls;
    
    // This matches the maze layout in Rendering.cpp createMazeWallsRAII()
    
    // Central vertical wall with gap in middle
    walls.emplace_back(0.f, halfH * 0.8f, 0.f, halfH * 0.2f, thickness);
    walls.emplace_back(0.f, -halfH * 0.2f, 0.f, -halfH * 0.8f, thickness);
    
    // Central horizontal wall with gap in middle
    walls.emplace_back(-halfW * 0.8f, 0.f, -halfW * 0.2f, 0.f, thickness);
    walls.emplace_back(halfW * 0.2f, 0.f, halfW * 0.8f, 0.f, thickness);
    
    // Extra obstacles
    walls.emplace_back(-halfW * 0.5f, halfH * 0.5f, halfW * 0.0f, halfH * 0.5f, thickness);
    walls.emplace_back(halfW * 0.0f, -halfH * 0.5f, halfW * 0.5f, -halfH * 0.5f, thickness);
    
    return walls;
}

std::vector<WallSegment> createBoundaryWalls(float halfW, float halfH, float thickness) {
    std::vector<WallSegment> walls;
    
    // Four boundary walls
    walls.emplace_back(-halfW, -halfH, halfW, -halfH, thickness);  // Bottom
    walls.emplace_back(halfW, -halfH, halfW, halfH, thickness);    // Right
    walls.emplace_back(halfW, halfH, -halfW, halfH, thickness);    // Top
    walls.emplace_back(-halfW, halfH, -halfW, -halfH, thickness);  // Left
    
    return walls;
}
