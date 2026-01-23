#include "Physics.hpp"
#include <cmath>
#include <algorithm>

PhysicsSimulator::PhysicsSimulator()
    : halfW(100.f)
    , halfH(140.f)
    , gravity(500.f)      // Gravity acceleration (units/s²)
    , friction(0.985f)    // Rolling friction (velocity retention per frame)
    , bounciness(0.4f)    // Bounce coefficient
    , maxVelocity(400.f)  // Maximum velocity cap
    , initialized(false)
{
}

void PhysicsSimulator::initialize(float halfW_, float halfH_, float ballRadius) {
    halfW = halfW_;
    halfH = halfH_;
    ball.radius = ballRadius;
    
    // Create boundary walls
    boundaryWalls.clear();
    float thick = 5.0f;
    boundaryWalls.emplace_back(-halfW, -halfH, halfW, -halfH, thick);  // Bottom
    boundaryWalls.emplace_back(halfW, -halfH, halfW, halfH, thick);    // Right
    boundaryWalls.emplace_back(halfW, halfH, -halfW, halfH, thick);    // Top
    boundaryWalls.emplace_back(-halfW, halfH, -halfW, -halfH, thick);  // Left
    
    reset();
    initialized = true;
}

void PhysicsSimulator::reset() {
    // Start ball in top-left corner
    ball.position = Vec2(-halfW * 0.75f, halfH * 0.75f);
    ball.velocity = Vec2(0.f, 0.f);
    currentTilt = Vec2(0.f, 0.f);
}

void PhysicsSimulator::addWall(const WallSegment& wall) {
    walls.push_back(wall);
}

void PhysicsSimulator::setGoal(const Goal& g) {
    goal = g;
}

void PhysicsSimulator::clearWalls() {
    walls.clear();
}

Vec2 PhysicsSimulator::calculateTilt(const cv::Mat& rvec) {
    if (rvec.empty()) return Vec2(0, 0);
    
    // Convert rotation vector to rotation matrix
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    
    // The Z column of the rotation matrix gives the surface normal
    double nx = R.at<double>(0, 2);
    double ny = R.at<double>(1, 2);
    double nz = R.at<double>(2, 2);
    
    // Calculate tilt angles from normal
    // tiltX affects ball movement in X direction
    // tiltY affects ball movement in Y direction
    float tiltX = (float)std::atan2(nx, std::abs(nz));
    float tiltY = (float)std::atan2(-ny, std::abs(nz));
    
    // Clamp tilt to reasonable values (max ~30 degrees)
    const float maxTilt = 0.52f;  // ~30 degrees in radians
    tiltX = std::clamp(tiltX, -maxTilt, maxTilt);
    tiltY = std::clamp(tiltY, -maxTilt, maxTilt);
    
    return Vec2(tiltX, tiltY);
}

Vec2 PhysicsSimulator::calculateGravityAcceleration(const Vec2& tilt) {
    // Project gravity onto the tilted plane
    // acceleration = g * sin(tilt_angle) for each axis
    float ax = gravity * std::sin(tilt.x);
    float ay = gravity * std::sin(tilt.y);
    
    return Vec2(ax, ay);
}

void PhysicsSimulator::update(const cv::Mat& rvec, float deltaTime) {
    if (!initialized) return;
    
    // Clamp deltaTime to prevent instability
    deltaTime = std::min(deltaTime, 0.05f);
    
    // Calculate tilt with smoothing
    Vec2 newTilt = calculateTilt(rvec);
    const float tiltSmoothing = 0.25f;
    currentTilt.x = currentTilt.x * (1.f - tiltSmoothing) + newTilt.x * tiltSmoothing;
    currentTilt.y = currentTilt.y * (1.f - tiltSmoothing) + newTilt.y * tiltSmoothing;
    
    // Calculate gravity acceleration based on tilt
    Vec2 acceleration = calculateGravityAcceleration(currentTilt);
    
    // Semi-implicit Euler integration
    // Update velocity first
    ball.velocity += acceleration * deltaTime;
    
    // Apply friction
    ball.velocity *= friction;
    
    // Cap velocity
    float speed = ball.velocity.length();
    if (speed > maxVelocity) {
        ball.velocity = ball.velocity.normalized() * maxVelocity;
    }
    
    // Update position
    ball.position += ball.velocity * deltaTime;
    
    // Resolve collisions
    resolveWallCollisions();
    resolveBoundaryCollisions();
}

Vec2 PhysicsSimulator::closestPointOnSegment(const Vec2& p, const Vec2& a, const Vec2& b) {
    Vec2 ab = b - a;
    float t = (p - a).dot(ab) / (ab.dot(ab) + 0.0001f);
    t = std::clamp(t, 0.f, 1.f);
    return Vec2(a.x + t * ab.x, a.y + t * ab.y);
}

bool PhysicsSimulator::checkWallCollision(const WallSegment& wall, Vec2& normal, float& penetration) {
    // Find closest point on wall to ball center
    Vec2 closest = closestPointOnSegment(ball.position, wall.p1, wall.p2);
    
    // Calculate distance
    Vec2 diff = ball.position - closest;
    float dist = diff.length();
    
    // Collision threshold
    float threshold = ball.radius + wall.thickness * 0.5f;
    
    if (dist < threshold && dist > 0.001f) {
        normal = diff * (1.f / dist);  // Normalize
        penetration = threshold - dist;
        return true;
    }
    
    return false;
}

void PhysicsSimulator::resolveWallCollisions() {
    const int maxIterations = 5;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        bool collisionFound = false;
        
        // Check maze walls
        for (const auto& wall : walls) {
            Vec2 normal;
            float penetration;
            
            if (checkWallCollision(wall, normal, penetration)) {
                collisionFound = true;
                
                // Push ball out of wall
                ball.position += normal * penetration;
                
                // Reflect velocity with bounce
                float normalVelocity = ball.velocity.dot(normal);
                if (normalVelocity < 0) {
                    ball.velocity = ball.velocity - normal * ((1.f + bounciness) * normalVelocity);
                    ball.velocity *= 0.9f;  // Extra friction on collision
                }
            }
        }
        
        if (!collisionFound) break;
    }
}

void PhysicsSimulator::resolveBoundaryCollisions() {
    float effectiveHalfW = halfW - ball.radius;
    float effectiveHalfH = halfH - ball.radius;
    
    // Left boundary
    if (ball.position.x < -effectiveHalfW) {
        ball.position.x = -effectiveHalfW;
        if (ball.velocity.x < 0) {
            ball.velocity.x = -ball.velocity.x * bounciness;
        }
    }
    
    // Right boundary
    if (ball.position.x > effectiveHalfW) {
        ball.position.x = effectiveHalfW;
        if (ball.velocity.x > 0) {
            ball.velocity.x = -ball.velocity.x * bounciness;
        }
    }
    
    // Bottom boundary
    if (ball.position.y < -effectiveHalfH) {
        ball.position.y = -effectiveHalfH;
        if (ball.velocity.y < 0) {
            ball.velocity.y = -ball.velocity.y * bounciness;
        }
    }
    
    // Top boundary
    if (ball.position.y > effectiveHalfH) {
        ball.position.y = effectiveHalfH;
        if (ball.velocity.y > 0) {
            ball.velocity.y = -ball.velocity.y * bounciness;
        }
    }
}

bool PhysicsSimulator::isAtGoal() const {
    float dx = ball.position.x - goal.position.x;
    float dy = ball.position.y - goal.position.y;
    float dist = std::sqrt(dx * dx + dy * dy);
    return dist < goal.radius - ball.radius * 0.5f;
}

bool PhysicsSimulator::isBallInHole() const {
    return isAtGoal();
}
