#include "Drawing.hpp"
#include <sstream>
#include <iomanip>

void drawLabeledQuad(cv::Mat& frame, const std::vector<cv::Point2f>& q) {
    if (q.size() != 4) return;
    
    cv::Scalar col = {0, 255, 255}; // Yellow
    
    // Draw edges
    for (int i = 0; i < 4; ++i) {
        cv::line(frame, q[i], q[(i + 1) % 4], col, 2, cv::LINE_AA);
    }
    
    // Draw corner markers with labels
    const char* labels[] = {"TL", "BL", "BR", "TR"};
    cv::Scalar colors[] = {
        {0, 0, 255},   // Red for TL
        {0, 255, 0},   // Green for BL 
        {255, 0, 0},   // Blue for BR
        {255, 255, 0}  // Cyan for TR 
    };
    
    for (int i = 0; i < 4; ++i) {
        cv::circle(frame, q[i], 8, colors[i], -1, cv::LINE_AA);
        cv::circle(frame, q[i], 8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        
        std::ostringstream ss;
        ss << labels[i] << " (" << (int)q[i].x << "," << (int)q[i].y << ")";
        cv::putText(frame, ss.str(), q[i] + cv::Point2f(10, -10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, ss.str(), q[i] + cv::Point2f(10, -10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1);
    }
}

void drawAxes(cv::Mat& frame, const cv::Mat& K, const cv::Mat& D,
              const cv::Mat& rvec, const cv::Mat& tvec, float L) {
    std::vector<cv::Point3f> obj = { 
        {0, 0, 0}, 
        {L, 0, 0}, 
        {0, L, 0}, 
        {0, 0, L}   // Positive Z (into surface for OpenCV convention)
    };
    
    std::vector<cv::Point2f> img;
    cv::projectPoints(obj, rvec, tvec, K, D, img);
    
    // X axis - Red
    cv::line(frame, img[0], img[1], {0, 0, 255}, 3, cv::LINE_AA);
    cv::putText(frame, "X", img[1] + cv::Point2f(5, -5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 255}, 2);
    
    // Y axis - Green
    cv::line(frame, img[0], img[2], {0, 255, 0}, 3, cv::LINE_AA);
    cv::putText(frame, "Y", img[2] + cv::Point2f(5, -5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
    
    // Z axis - Blue
    cv::line(frame, img[0], img[3], {255, 0, 0}, 3, cv::LINE_AA);
    cv::putText(frame, "Z", img[3] + cv::Point2f(5, -5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 0, 0}, 2);
}

void drawCubeWireframe(cv::Mat& frame, const std::vector<cv::Point2f>& imgPts) {
    if (imgPts.size() != 8) return;
    
    auto line = [&](int i, int j) {
        cv::line(frame, imgPts[i], imgPts[j], {0, 255, 255}, 2, cv::LINE_AA);
    };
    
    // Bottom face
    line(0, 1); line(1, 2); line(2, 3); line(3, 0);
    // Top face
    line(4, 5); line(5, 6); line(6, 7); line(7, 4);
    // Vertical edges
    line(0, 4); line(1, 5); line(2, 6); line(3, 7);
}

void drawCubeFaces(cv::Mat& frame,
                   const std::vector<cv::Point3f>& cubePts3D,
                   const cv::Mat& rvec, const cv::Mat& tvec,
                   const cv::Mat& K, const cv::Mat& D) {
    if (cubePts3D.size() != 8) return;
    
    // Project 3D points to 2D
    std::vector<cv::Point2f> pts2D;
    cv::projectPoints(cubePts3D, rvec, tvec, K, D, pts2D);

    // Convert rotation vector to matrix
    cv::Matx33d R;
    cv::Rodrigues(rvec, R);
    cv::Vec3d tv(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    // Transform points to camera coordinates
    std::vector<cv::Vec3d> camPts(8);
    for (int i = 0; i < 8; ++i) {
        cv::Vec3d X(cubePts3D[i].x, cubePts3D[i].y, cubePts3D[i].z);
        camPts[i] = R * X + tv;
    }
    
    // Face definitions (vertex indices)
    const int F[6][4] = {
        {0, 1, 2, 3},  // Bottom
        {4, 5, 6, 7},  // Top
        {0, 1, 5, 4},  // Front
        {1, 2, 6, 5},  // Right
        {2, 3, 7, 6},  // Back
        {3, 0, 4, 7}   // Left
    };
    
    // Face colors
    const cv::Scalar cols[6] = {
        {60, 60, 200},    // Red-ish
        {60, 200, 60},    // Green-ish
        {200, 60, 60},    // Blue-ish
        {200, 200, 60},   // Cyan-ish
        {200, 60, 200},   // Magenta-ish
        {60, 200, 200}    // Yellow-ish
    };
    
    // Determine visible faces and sort by depth
    struct Face { int idx; double zavg; };
    std::vector<Face> vis;
    
    for (int f = 0; f < 6; ++f) {
        cv::Vec3d a = camPts[F[f][0]], b = camPts[F[f][1]], c = camPts[F[f][2]];
        cv::Vec3d v1 = b - a, v2 = c - b;
        
        // Calculate face normal
        cv::Vec3d N(v1[1] * v2[2] - v1[2] * v2[1],
                    v1[2] * v2[0] - v1[0] * v2[2],
                    v1[0] * v2[1] - v1[1] * v2[0]);
        
        // Calculate average Z depth
        double z = 0;
        for (int k = 0; k < 4; ++k) {
            z += camPts[F[f][k]][2];
        }
        z /= 4.0;
        
        // Face is visible if normal points toward camera (negative Z component)
        if (N[2] < 0.0) {
            vis.push_back({f, z});
        }
    }
    
    // Sort faces by depth (far to near)
    std::sort(vis.begin(), vis.end(), [](const Face& a, const Face& b) {
        return a.zavg > b.zavg;
    });

    const double opacity = 0.85;

    // Draw visible faces
    for (auto& f : vis) {
        std::vector<cv::Point> poly;
        poly.reserve(4);
        for (int k = 0; k < 4; ++k) {
            poly.emplace_back(pts2D[F[f.idx][k]]);
        }
        
        cv::Mat overlay = frame.clone();
        cv::fillConvexPoly(overlay, poly, cols[f.idx], cv::LINE_AA);
        cv::addWeighted(overlay, opacity, frame, 1.0 - opacity, 0.0, frame);
        cv::polylines(frame, poly, true, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
    }
}

void drawStatusOverlay(cv::Mat& frame, bool detected, double confidence,
                       const std::string& status) {
    // Background rectangle
    cv::rectangle(frame, cv::Rect(10, 10, 250, 80), cv::Scalar(0, 0, 0, 128), -1);
    
    // Detection status
    cv::Scalar statusColor = detected ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    std::string detText = detected ? "DETECTED" : "NOT DETECTED";
    cv::putText(frame, detText, cv::Point(20, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);
    
    // Confidence bar
    cv::rectangle(frame, cv::Rect(20, 45, 200, 15), cv::Scalar(50, 50, 50), -1);
    int barWidth = (int)(200 * confidence);
    cv::Scalar barColor = confidence > 0.7 ? cv::Scalar(0, 255, 0) : 
                          confidence > 0.4 ? cv::Scalar(0, 255, 255) : 
                          cv::Scalar(0, 0, 255);
    cv::rectangle(frame, cv::Rect(20, 45, barWidth, 15), barColor, -1);
    
    // Confidence percentage
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << (confidence * 100) << "%";
    cv::putText(frame, ss.str(), cv::Point(225, 58),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    // Status message
    cv::putText(frame, status, cv::Point(20, 80),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
}