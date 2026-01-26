#include "Detection.hpp"
#include <algorithm>
#include <cmath>

std::vector<cv::Point2f> orderQuadRobust(const std::vector<cv::Point2f>& quadRaw) {
    if (quadRaw.size() != 4) return {};
    
    std::vector<cv::Point2f> P = quadRaw;
    
    // Calculate centroid
    cv::Point2f c(0.f, 0.f);
    for (auto& p : P) c += p;
    c *= 0.25f;
    
    // Sort by angle from centroid
    std::sort(P.begin(), P.end(), [&](const cv::Point2f& a, const cv::Point2f& b) {
        float aa = std::atan2(a.y - c.y, a.x - c.x);
        float bb = std::atan2(b.y - c.y, b.x - c.x);
        return aa < bb;
    });
    
    // Calculate signed area to determine winding order
    auto signedArea = [&](const std::vector<cv::Point2f>& Q) {
        double A = 0;
        for (int i = 0; i < 4; ++i) {
            auto& u = Q[i];
            auto& v = Q[(i + 1) % 4];
            A += u.x * v.y - v.x * u.y;
        }
        return A;
    };
    
    // Ensure counter-clockwise ordering
    if (signedArea(P) > 0.0) {
        std::reverse(P.begin() + 1, P.end());
    }
    
    // Find top-left corner (minimum x + y)
    int tl = 0;
    float best = P[0].x + P[0].y;
    for (int i = 1; i < 4; ++i) {
        float s = P[i].x + P[i].y;
        if (s < best) {
            best = s;
            tl = i;
        }
    }
    
    // Rotate array so top-left is first
    std::vector<cv::Point2f> O(4);
    for (int i = 0; i < 4; ++i) {
        O[i] = P[(tl + i) % 4];
    }
    
    return O;
}

bool findLargestQuad(const cv::Mat& gray, std::vector<cv::Point2f>& quadOrdered) {
    // Apply Gaussian blur and edge detection
    cv::Mat blur, edges;
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);
    cv::Canny(blur, edges, 75, 200);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find largest quadrilateral
    double maxArea = 0.0;
    std::vector<cv::Point> best;
    
    for (auto& c : contours) {
        if (c.size() < 4) continue;
        
        double peri = cv::arcLength(c, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.008 * peri, true);
        
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            double area = std::fabs(cv::contourArea(approx));
            if (area > maxArea) {
                maxArea = area;
                best = approx;
            }
        }
    }
    
    if (best.empty()) return false;

    // Convert to Point2f and refine corners
    cv::Mat ptsf(4, 1, CV_32FC2);
    for (int i = 0; i < 4; ++i) {
        ptsf.at<cv::Point2f>(i, 0) = best[i];
    }

    // Ensure grayscale is 8-bit for cornerSubPix
    cv::Mat gray8;
    if (gray.type() != CV_8U) {
        gray.convertTo(gray8, CV_8U);
    } else {
        gray8 = gray;
    }
    
    // Refine corner positions
    cv::cornerSubPix(gray8, ptsf, cv::Size(3, 3), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 40, 0.001));

    // Extract refined points
    std::vector<cv::Point2f> q4(4);
    for (int i = 0; i < 4; ++i) {
        q4[i] = ptsf.at<cv::Point2f>(i, 0);
    }
    
    // Order the points
    quadOrdered = orderQuadRobust(q4);
    return quadOrdered.size() == 4;
}

bool detectSheetAdaptive(const cv::Mat& frame, 
                         std::vector<cv::Point2f>& corners,
                         double& confidence) {
    corners.clear();
    confidence = 0.0;
    
    if (frame.empty()) return false;

    // Convert to grayscale
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = frame.clone();
    }

    // Calculate image statistics for adaptive parameters
    cv::Scalar meanScalar, stdScalar;
    cv::meanStdDev(gray, meanScalar, stdScalar);
    double meanVal = meanScalar[0];
    double stdVal = stdScalar[0];

    // Adaptive blur kernel size based on texture
    int blurK = std::clamp((int)((50 - std::min(stdVal, 50.0)) / 10.0) * 2 + 3, 3, 9);
    if (blurK % 2 == 0) blurK++;
    cv::GaussianBlur(gray, gray, cv::Size(blurK, blurK), 0);

    // Apply CLAHE for low contrast images
    if (stdVal < 25.0) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(3.0);
        clahe->apply(gray, gray);
        cv::meanStdDev(gray, meanScalar, stdScalar);
        stdVal = stdScalar[0];
    }

    // LOWERED Canny thresholds for more continuous detection
    double baseLow = 25.0;    // Was 30.0
    double baseHigh = 100.0;  // Was 120.0
    double textureFactor = std::clamp(stdVal / 40.0, 0.4, 1.6);
    double exposureFactor = std::clamp(meanVal / 128.0, 0.6, 1.4);
    double cannyLow = std::clamp(baseLow * textureFactor * exposureFactor, 18.0, 75.0);   // Lowered min from 20
    double cannyHigh = std::clamp(baseHigh * textureFactor * exposureFactor, 75.0, 210.0); // Lowered min from 80

    cv::Mat edges;
    cv::Canny(gray, edges, cannyLow, cannyHigh);

    // Check edge density
    double edgeRatio = (double)cv::countNonZero(edges) / (edges.rows * edges.cols);
    
    // LOWERED threshold for Otsu fallback
    if (edgeRatio < 0.004) {  // Was 0.005
        cv::Mat otsu;
        cv::threshold(gray, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::Canny(otsu, edges, cannyLow * 0.5, cannyHigh * 0.5);
    }

    // Morphological closing to connect edges
    int morphK = (edgeRatio < 0.015) ? 7 : (edgeRatio > 0.04 ? 3 : 5);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morphK, morphK));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);

    // Find contours - EXTERNAL only (ignores internal rectangles)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double frameArea = (double)frame.cols * frame.rows;
    double bestScore = 0.0;
    std::vector<cv::Point> bestPoly;

    // A4 aspect ratio
    const double A4_AR = 297.0 / 210.0;

    for (auto& c : contours) {
        double peri = cv::arcLength(c, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.02 * peri, true);
        
        if (approx.size() != 4 || !cv::isContourConvex(approx)) continue;
        
        double area = std::fabs(cv::contourArea(approx));
        
        // LOWERED minimum area for more continuous detection
        if (area < 0.0015 * frameArea) continue;  // Was 0.002
        
        cv::Rect br = cv::boundingRect(approx);
        double w = br.width, h = br.height;
        if (w < 10 || h < 10) continue;
        
        // Check aspect ratio - RELAXED tolerance
        double ar = std::max(w, h) / std::max(1.0, std::min(w, h));
        double arErr = std::fabs(ar - A4_AR) / A4_AR;
        if (arErr > 0.45) continue;  // Was 0.35 - more lenient

        // Score based on AREA (biggest wins) and aspect ratio match
        // Area heavily weighted - ensures LARGEST rectangle is chosen
        double score = area * (1.0 - arErr * 0.3);  // Reduced AR penalty from 1.0 to 0.3
        if (score > bestScore) {
            bestScore = score;
            bestPoly = approx;
        }
    }

    if (bestPoly.empty()) {
        return false;
    }

    // Convert to Point2f and order
    std::vector<cv::Point2f> rawPts;
    for (const auto& p : bestPoly) {
        rawPts.push_back(cv::Point2f((float)p.x, (float)p.y));
    }
    
    std::vector<cv::Point2f> ordered = orderQuadRobust(rawPts);
    
    // Refine corners
    cv::cornerSubPix(gray, ordered, cv::Size(5, 5), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 25, 0.01));
    
    corners = ordered;

    // MORE GENEROUS confidence calculation
    double textureScore = std::clamp(stdVal / 60.0, 0.4, 1.0);  // Raised min from 0.3
    double areaNorm = bestScore / frameArea;
    double areaScore = std::clamp(areaNorm / 0.12, 0.0, 1.0);   // Lowered threshold from 0.15
    double edgeScore = std::clamp(edgeRatio / 0.035, 0.3, 1.0); // Lowered threshold, raised min
    confidence = 0.60 + 0.12 * textureScore + 0.15 * areaScore + 0.13 * edgeScore;  // Higher base

    return true;
}