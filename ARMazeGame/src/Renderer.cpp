#include "Renderer.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>

MazeRenderer::MazeRenderer()
    : wallHeight(25.0f)
    , wallColor(100, 100, 100)
    , ballColor(50, 50, 220)
    , goalColor(50, 200, 50)
    , planeColor(200, 180, 150)
    , boundaryColor(80, 80, 80)
{
}

std::vector<cv::Point2f> MazeRenderer::projectPoints3D(const std::vector<cv::Point3f>& points3D,
                                                        const cv::Mat& K, const cv::Mat& D,
                                                        const cv::Mat& rvec, const cv::Mat& tvec) {
    std::vector<cv::Point2f> points2D;
    if (!points3D.empty()) {
        cv::projectPoints(points3D, rvec, tvec, K, D, points2D);
    }
    return points2D;
}

void MazeRenderer::drawFilledPolygon(cv::Mat& frame, const std::vector<cv::Point2f>& points,
                                      const cv::Scalar& color, double alpha) {
    if (points.size() < 3) return;
    
    std::vector<cv::Point> intPoints;
    for (const auto& p : points) {
        intPoints.emplace_back((int)p.x, (int)p.y);
    }
    
    cv::Mat overlay = frame.clone();
    cv::fillConvexPoly(overlay, intPoints, color, cv::LINE_AA);
    cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame);
}

void MazeRenderer::drawWallSegment3D(cv::Mat& frame,
                                      const cv::Mat& K, const cv::Mat& D,
                                      const cv::Mat& rvec, const cv::Mat& tvec,
                                      const WallSegment& wall, float height,
                                      const cv::Scalar& color) {
    // Get wall direction and perpendicular
    Vec2 dir(wall.p2.x - wall.p1.x, wall.p2.y - wall.p1.y);
    float len = dir.length();
    if (len < 0.001f) return;
    
    Vec2 norm(-dir.y / len, dir.x / len);
    float halfThick = wall.thickness * 0.5f;
    
    // Calculate 8 corners of the wall box
    std::vector<cv::Point3f> corners3D = {
        // Bottom face
        cv::Point3f(wall.p1.x - norm.x * halfThick, wall.p1.y - norm.y * halfThick, 0),
        cv::Point3f(wall.p1.x + norm.x * halfThick, wall.p1.y + norm.y * halfThick, 0),
        cv::Point3f(wall.p2.x + norm.x * halfThick, wall.p2.y + norm.y * halfThick, 0),
        cv::Point3f(wall.p2.x - norm.x * halfThick, wall.p2.y - norm.y * halfThick, 0),
        // Top face (Z is up/out of plane in our coordinate system)
        cv::Point3f(wall.p1.x - norm.x * halfThick, wall.p1.y - norm.y * halfThick, height),
        cv::Point3f(wall.p1.x + norm.x * halfThick, wall.p1.y + norm.y * halfThick, height),
        cv::Point3f(wall.p2.x + norm.x * halfThick, wall.p2.y + norm.y * halfThick, height),
        cv::Point3f(wall.p2.x - norm.x * halfThick, wall.p2.y - norm.y * halfThick, height)
    };
    
    // Project to 2D
    std::vector<cv::Point2f> corners2D = projectPoints3D(corners3D, K, D, rvec, tvec);
    if (corners2D.size() != 8) return;
    
    // Draw visible faces (simplified - draw all faces with depth sorting implied by draw order)
    // Face definitions: bottom, top, front, back, left, right
    int faces[6][4] = {
        {0, 1, 2, 3},  // Bottom
        {4, 5, 6, 7},  // Top
        {0, 1, 5, 4},  // Front
        {2, 3, 7, 6},  // Back
        {0, 3, 7, 4},  // Left
        {1, 2, 6, 5}   // Right
    };
    
    // Calculate which faces are visible based on normal direction to camera
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Vec3d camDir(R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));
    
    // Draw faces with slight transparency
    for (int f = 0; f < 6; ++f) {
        std::vector<cv::Point2f> facePoints = {
            corners2D[faces[f][0]],
            corners2D[faces[f][1]],
            corners2D[faces[f][2]],
            corners2D[faces[f][3]]
        };
        
        // Calculate face center Z for depth sorting
        cv::Scalar faceColor = color;
        if (f == 1) {  // Top face - slightly brighter
            faceColor = cv::Scalar(color[0] * 1.2, color[1] * 1.2, color[2] * 1.2);
        } else if (f == 0) {  // Bottom face - slightly darker
            faceColor = cv::Scalar(color[0] * 0.8, color[1] * 0.8, color[2] * 0.8);
        }
        
        drawFilledPolygon(frame, facePoints, faceColor, 0.85);
    }
    
    // Draw edges for better definition
    for (int i = 0; i < 4; ++i) {
        // Bottom edges
        cv::line(frame, corners2D[i], corners2D[(i + 1) % 4], cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
        // Top edges
        cv::line(frame, corners2D[i + 4], corners2D[((i + 1) % 4) + 4], cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
        // Vertical edges
        cv::line(frame, corners2D[i], corners2D[i + 4], cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
    }
}

void MazeRenderer::render(cv::Mat& frame,
                          const cv::Mat& K, const cv::Mat& D,
                          const cv::Mat& rvec, const cv::Mat& tvec,
                          const PhysicsSimulator& physics,
                          const GameState& gameState) {
    // Get dimensions from physics
    float halfW = 100.0f;  // Will be set based on actual detection
    float halfH = 140.0f;
    
    // Render plane outline
    renderPlaneOutline(frame, K, D, rvec, tvec, halfW, halfH);
    
    // Render boundary walls
    renderWalls(frame, K, D, rvec, tvec, physics.getBoundaryWalls(), wallHeight, boundaryColor);
    
    // Render maze walls
    renderWalls(frame, K, D, rvec, tvec, physics.getWalls(), wallHeight, wallColor);
    
    // Render goal
    renderGoal(frame, K, D, rvec, tvec, physics.getGoal());
    
    // Render ball with velocity arrows
    renderBall(frame, K, D, rvec, tvec, physics.getBallPosition(), physics.getBallRadius(), physics.getBallVelocity());
    
    // Render HUD
    renderHUD(frame, gameState, physics.getTilt(), true);
    
    // Render win screen if won
    if (gameState.getState() == GameState::State::WIN) {
        renderWinScreen(frame, gameState);
    }
}

void MazeRenderer::renderPlaneOutline(cv::Mat& frame,
                                       const cv::Mat& K, const cv::Mat& D,
                                       const cv::Mat& rvec, const cv::Mat& tvec,
                                       float halfW, float halfH) {
    // Define plane corners
    std::vector<cv::Point3f> planeCorners = {
        cv::Point3f(-halfW, -halfH, 0),
        cv::Point3f(halfW, -halfH, 0),
        cv::Point3f(halfW, halfH, 0),
        cv::Point3f(-halfW, halfH, 0)
    };
    
    std::vector<cv::Point2f> corners2D = projectPoints3D(planeCorners, K, D, rvec, tvec);
    
    if (corners2D.size() == 4) {
        // Draw filled plane with transparency
        drawFilledPolygon(frame, corners2D, planeColor, 0.3);
        
        // Draw outline
        for (int i = 0; i < 4; ++i) {
            cv::line(frame, corners2D[i], corners2D[(i + 1) % 4], 
                    cv::Scalar(0, 200, 0), 2, cv::LINE_AA);
        }
    }
}

void MazeRenderer::renderWalls(cv::Mat& frame,
                                const cv::Mat& K, const cv::Mat& D,
                                const cv::Mat& rvec, const cv::Mat& tvec,
                                const std::vector<WallSegment>& walls,
                                float height,
                                const cv::Scalar& color) {
    for (const auto& wall : walls) {
        drawWallSegment3D(frame, K, D, rvec, tvec, wall, height, color);
    }
}

void MazeRenderer::renderBall(cv::Mat& frame,
                               const cv::Mat& K, const cv::Mat& D,
                               const cv::Mat& rvec, const cv::Mat& tvec,
                               const Vec2& position, float radius,
                               const Vec2& velocity) {
    // Ball is rendered as a filled circle at its position
    // Ball center is at Z = radius (sitting on the plane)
    cv::Point3f ballCenter(position.x, position.y, radius);
    
    std::vector<cv::Point3f> ballPoint = {ballCenter};
    std::vector<cv::Point2f> ball2D = projectPoints3D(ballPoint, K, D, rvec, tvec);
    
    if (!ball2D.empty()) {
        // Estimate ball radius in pixels by projecting a point at ball edge
        cv::Point3f ballEdge(position.x + radius, position.y, radius);
        std::vector<cv::Point3f> edgePoint = {ballEdge};
        std::vector<cv::Point2f> edge2D = projectPoints3D(edgePoint, K, D, rvec, tvec);
        
        int pixelRadius = 15;  // Default
        if (!edge2D.empty()) {
            pixelRadius = (int)cv::norm(ball2D[0] - edge2D[0]);
            pixelRadius = std::max(5, std::min(50, pixelRadius));
        }
        
        cv::Point center((int)ball2D[0].x, (int)ball2D[0].y);
        
        // Draw velocity arrows BEFORE the ball so they appear behind/beside it
        float velScale = 0.15f;  // Scale factor for arrow length
        float minVelForArrow = 10.0f;  // Minimum velocity to show arrow
        
        // X velocity arrow (Red)
        if (std::abs(velocity.x) > minVelForArrow) {
            cv::Point3f arrowEnd(position.x + velocity.x * velScale, position.y, radius);
            std::vector<cv::Point3f> arrowPt = {arrowEnd};
            std::vector<cv::Point2f> arrow2D = projectPoints3D(arrowPt, K, D, rvec, tvec);
            
            if (!arrow2D.empty()) {
                cv::Scalar xColor(0, 0, 255);  // Red for X
                cv::arrowedLine(frame, center, cv::Point((int)arrow2D[0].x, (int)arrow2D[0].y),
                               xColor, 2, cv::LINE_AA, 0, 0.3);
            }
        }
        
        // Y velocity arrow (Green)
        if (std::abs(velocity.y) > minVelForArrow) {
            cv::Point3f arrowEnd(position.x, position.y + velocity.y * velScale, radius);
            std::vector<cv::Point3f> arrowPt = {arrowEnd};
            std::vector<cv::Point2f> arrow2D = projectPoints3D(arrowPt, K, D, rvec, tvec);
            
            if (!arrow2D.empty()) {
                cv::Scalar yColor(0, 255, 0);  // Green for Y
                cv::arrowedLine(frame, center, cv::Point((int)arrow2D[0].x, (int)arrow2D[0].y),
                               yColor, 2, cv::LINE_AA, 0, 0.3);
            }
        }
        
        // Combined velocity arrow (Cyan) - shows actual direction
        float speed = velocity.length();
        if (speed > minVelForArrow) {
            cv::Point3f arrowEnd(position.x + velocity.x * velScale, 
                                position.y + velocity.y * velScale, radius);
            std::vector<cv::Point3f> arrowPt = {arrowEnd};
            std::vector<cv::Point2f> arrow2D = projectPoints3D(arrowPt, K, D, rvec, tvec);
            
            if (!arrow2D.empty()) {
                cv::Scalar combinedColor(255, 255, 0);  // Cyan for combined
                cv::arrowedLine(frame, center, cv::Point((int)arrow2D[0].x, (int)arrow2D[0].y),
                               combinedColor, 3, cv::LINE_AA, 0, 0.25);
            }
        }
        
        // Outer shadow
        cv::circle(frame, cv::Point(center.x + 2, center.y + 2), pixelRadius, 
                  cv::Scalar(30, 30, 30), -1, cv::LINE_AA);
        
        // Main ball
        cv::circle(frame, center, pixelRadius, ballColor, -1, cv::LINE_AA);
        
        // Highlight
        cv::circle(frame, cv::Point(center.x - pixelRadius/3, center.y - pixelRadius/3), 
                  pixelRadius/3, cv::Scalar(100, 100, 255), -1, cv::LINE_AA);
        
        // Outline
        cv::circle(frame, center, pixelRadius, cv::Scalar(20, 20, 100), 2, cv::LINE_AA);
    }
}

void MazeRenderer::renderGoal(cv::Mat& frame,
                               const cv::Mat& K, const cv::Mat& D,
                               const cv::Mat& rvec, const cv::Mat& tvec,
                               const Goal& goal) {
    // Draw goal as a circle on the plane
    const int segments = 24;
    std::vector<cv::Point3f> goalPoints;
    
    for (int i = 0; i <= segments; ++i) {
        float angle = (float)i / segments * 2.0f * CV_PI;
        float x = goal.position.x + goal.radius * std::cos(angle);
        float y = goal.position.y + goal.radius * std::sin(angle);
        goalPoints.emplace_back(x, y, 0.1f);  // Slightly above plane
    }
    
    std::vector<cv::Point2f> goal2D = projectPoints3D(goalPoints, K, D, rvec, tvec);
    
    if (goal2D.size() > 2) {
        // Draw filled goal
        std::vector<cv::Point> intPoints;
        for (const auto& p : goal2D) {
            intPoints.emplace_back((int)p.x, (int)p.y);
        }
        
        cv::Mat overlay = frame.clone();
        cv::fillPoly(overlay, intPoints, goalColor, cv::LINE_AA);
        cv::addWeighted(overlay, 0.6, frame, 0.4, 0.0, frame);
        
        // Draw outline
        cv::polylines(frame, intPoints, true, cv::Scalar(0, 100, 0), 2, cv::LINE_AA);
        
        // Draw "GOAL" text at center
        cv::Point3f goalCenter(goal.position.x, goal.position.y, 0.2f);
        std::vector<cv::Point3f> centerPoint = {goalCenter};
        std::vector<cv::Point2f> center2D = projectPoints3D(centerPoint, K, D, rvec, tvec);
        if (!center2D.empty()) {
            cv::putText(frame, "GOAL", 
                       cv::Point((int)center2D[0].x - 20, (int)center2D[0].y + 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
}

void MazeRenderer::renderHUD(cv::Mat& frame, const GameState& gameState,
                              const Vec2& tilt, bool detected) {
    int y = 20;
    
    // Background panel
    cv::rectangle(frame, cv::Rect(5, 5, 200, 90), cv::Scalar(0, 0, 0), -1);
    cv::rectangle(frame, cv::Rect(5, 5, 200, 90), cv::Scalar(100, 100, 100), 1);
    
    // Detection status
    cv::Scalar statusColor = detected ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    std::string statusText = detected ? "TRACKING" : "SEARCHING...";
    cv::putText(frame, statusText, cv::Point(10, y + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, statusColor, 1, cv::LINE_AA);
    
    // Timer
    y += 25;
    std::ostringstream timeStr;
    timeStr << "Time: " << std::fixed << std::setprecision(1) << gameState.getElapsedTime() << "s";
    cv::putText(frame, timeStr.str(), cv::Point(10, y + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    
    // Attempts
    y += 20;
    std::ostringstream attStr;
    attStr << "Attempts: " << gameState.getAttempts();
    cv::putText(frame, attStr.str(), cv::Point(10, y + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    
    // Tilt indicator
    y += 20;
    cv::putText(frame, "Tilt:", cv::Point(10, y + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    
    // Draw tilt indicator
    int indicatorX = 150;
    int indicatorY = y + 5;
    int indicatorSize = 30;
    
    cv::circle(frame, cv::Point(indicatorX, indicatorY), indicatorSize, cv::Scalar(50, 50, 50), -1);
    cv::circle(frame, cv::Point(indicatorX, indicatorY), indicatorSize, cv::Scalar(150, 150, 150), 1);
    
    // Draw tilt direction
    int dx = (int)(tilt.x * indicatorSize * 2);
    int dy = (int)(-tilt.y * indicatorSize * 2);
    dx = std::clamp(dx, -indicatorSize + 3, indicatorSize - 3);
    dy = std::clamp(dy, -indicatorSize + 3, indicatorSize - 3);
    
    cv::circle(frame, cv::Point(indicatorX + dx, indicatorY + dy), 5, cv::Scalar(0, 200, 255), -1);
    
    // Instructions at bottom
    int bottomY = frame.rows - 30;
    cv::rectangle(frame, cv::Rect(0, bottomY - 5, frame.cols, 35), cv::Scalar(0, 0, 0, 128), -1);
    cv::putText(frame, "Tilt the surface to roll the ball to the GOAL | Press 'R' to reset | 'Q' to quit",
                cv::Point(10, bottomY + 15), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
}

void MazeRenderer::renderWinScreen(cv::Mat& frame, const GameState& gameState) {
    // Semi-transparent overlay
    cv::Mat overlay = frame.clone();
    overlay = cv::Scalar(0, 100, 0);
    cv::addWeighted(overlay, 0.5, frame, 0.5, 0, frame);
    
    // Win text
    int centerX = frame.cols / 2;
    int centerY = frame.rows / 2;
    
    cv::putText(frame, "YOU WIN!", cv::Point(centerX - 120, centerY - 30),
                cv::FONT_HERSHEY_DUPLEX, 2.0, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
    
    std::ostringstream timeStr;
    timeStr << "Time: " << std::fixed << std::setprecision(2) << gameState.getElapsedTime() << " seconds";
    cv::putText(frame, timeStr.str(), cv::Point(centerX - 100, centerY + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    
    cv::putText(frame, "Press 'R' to play again or 'Q' to quit", 
                cv::Point(centerX - 160, centerY + 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
}