/**
 * @file Detection.cpp
 * @brief Implementation of A4 sheet detection algorithms
 */

#include "Detection.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

// Global adaptive params snapshot
static AdaptiveParamSnapshot g_lastAdaptiveParams;

const AdaptiveParamSnapshot& getLastAdaptiveParams() {
    return g_lastAdaptiveParams;
}

std::vector<cv::Point2f> orderQuadPoints(const std::vector<cv::Point>& pts) {
    std::vector<cv::Point2f> out;
    if (pts.size() != 4) return out;
    cv::Point2f center(0, 0);
    for (auto& p : pts) {
        out.push_back(cv::Point2f((float)p.x, (float)p.y));
        center += cv::Point2f((float)p.x, (float)p.y);
    }
    center *= 0.25f;
    std::sort(out.begin(), out.end(), [&](const cv::Point2f& a, const cv::Point2f& b) {
        return atan2(a.y - center.y, a.x - center.x) < atan2(b.y - center.y, b.x - center.x);
    });
    return out;
}

std::vector<cv::Point2f> orderQuadRobust(const std::vector<cv::Point2f>& quadRaw) {
    std::vector<cv::Point2f> P = quadRaw;
    if (P.size() != 4) return {};
    cv::Point2f c(0.f, 0.f);
    for (auto& p : P) c += p;
    c *= 0.25f;
    std::sort(P.begin(), P.end(), [&](const cv::Point2f& a, const cv::Point2f& b) {
        float aa = std::atan2(a.y - c.y, a.x - c.x);
        float bb = std::atan2(b.y - c.y, b.x - c.x);
        return aa < bb;
    });
    auto signedArea = [&](const std::vector<cv::Point2f>& Q) {
        double A = 0;
        for (int i = 0; i < 4; ++i) {
            auto& u = Q[i];
            auto& v = Q[(i + 1) % 4];
            A += u.x * v.y - v.x * u.y;
        }
        return A;
    };
    if (signedArea(P) > 0.0) std::reverse(P.begin() + 1, P.end());
    int tl = 0;
    float best = P[0].x + P[0].y;
    for (int i = 1; i < 4; ++i) {
        float s = P[i].x + P[i].y;
        if (s < best) {
            best = s;
            tl = i;
        }
    }
    std::vector<cv::Point2f> O(4);
    for (int i = 0; i < 4; ++i) O[i] = P[(tl + i) % 4];
    return O;
}

bool detectA4CornersStrict(const cv::Mat& frame, std::vector<cv::Point2f>& corners) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double maxArea = 0;
    std::vector<cv::Point> best;
    for (auto& c : contours) {
        double peri = cv::arcLength(c, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.02 * peri, true);
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            double area = fabs(cv::contourArea(approx));
            double frameArea = (double)frame.cols * frame.rows;
            if (area < 0.002 * frameArea) continue;
            cv::Rect br = cv::boundingRect(approx);
            double w = br.width, h = br.height;
            if (w < 5 || h < 5) continue;
            double ar = std::max(w, h) / std::max(1.0, std::min(w, h));
            const double A4_AR = 297.0 / 210.0;
            if (std::abs(ar - A4_AR) / A4_AR > 0.6) continue;
            if (area > maxArea) {
                maxArea = area;
                best = approx;
            }
        }
    }
    if (best.empty()) return false;
    std::vector<cv::Point2f> pts = orderQuadPoints(best);
    if (pts.size() != 4) return false;
    cv::cornerSubPix(gray, pts, cv::Size(5, 5), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01));
    corners = pts;
    return true;
}

bool detectA4CornersAdaptive(const cv::Mat& frame,
                             std::vector<cv::Point2f>& corners,
                             double& outConfidence)
{
    corners.clear();
    outConfidence = 0.0;
    if (frame.empty()) {
        std::cerr << "detectA4CornersAdaptive: empty frame\n";
        return false;
    }

    cv::Mat gray;
    if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else if (frame.channels() == 4) cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    else gray = frame.clone();

    cv::Scalar meanScalar, stdScalar;
    cv::meanStdDev(gray, meanScalar, stdScalar);
    double meanVal = meanScalar[0];
    double stdVal = stdScalar[0];

    int blurK = (std::clamp((int)((50 - std::min(stdVal, 50.0)) / 10.0) * 2 + 3, 3, 9));
    if (blurK % 2 == 0) blurK++;
    cv::GaussianBlur(gray, gray, cv::Size(blurK, blurK), 0);

    if (stdVal < 25.0) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(3.0);
        clahe->apply(gray, gray);
        cv::meanStdDev(gray, meanScalar, stdScalar);
        stdVal = stdScalar[0];
    }

    double baseLow = 30.0;
    double baseHigh = 120.0;
    double textureFactor = std::clamp(stdVal / 40.0, 0.4, 1.6);
    double exposureFactor = std::clamp((meanVal / 128.0), 0.6, 1.4);
    double cannyLow = baseLow * textureFactor * exposureFactor;
    double cannyHigh = baseHigh * textureFactor * exposureFactor;
    cannyLow = std::clamp(cannyLow, 20.0, 80.0);
    cannyHigh = std::clamp(cannyHigh, 80.0, 220.0);

    cv::Mat edges;
    cv::Canny(gray, edges, cannyLow, cannyHigh);

    double edgeRatio = (double)cv::countNonZero(edges) / (edges.rows * edges.cols);
    if (edgeRatio < 0.005) {
        cv::Mat otsu;
        cv::threshold(gray, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::Canny(otsu, edges, cannyLow * 0.5, cannyHigh * 0.5);
    }

    int morphK = (edgeRatio < 0.015) ? 7 : (edgeRatio > 0.04 ? 3 : 5);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morphK, morphK));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double frameArea = (double)frame.cols * frame.rows;
    double bestArea = 0.0;
    std::vector<cv::Point> bestPoly;

    for (auto& c : contours) {
        double peri = cv::arcLength(c, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.02 * peri, true);
        if (approx.size() != 4 || !cv::isContourConvex(approx)) continue;
        double area = std::fabs(cv::contourArea(approx));
        if (area < 0.002 * frameArea) continue;
        cv::Rect br = cv::boundingRect(approx);
        double w = br.width, h = br.height;
        if (w < 10 || h < 10) continue;
        double ar = std::max(w, h) / std::max(1.0, std::min(w, h));
        const double A4_AR = 297.0 / 210.0;
        double arErr = std::fabs(ar - A4_AR) / A4_AR;
        if (arErr > 0.35) continue;

        double score = area * (1.0 - arErr);
        if (score > bestArea) {
            bestArea = score;
            bestPoly = approx;
        }
    }

    if (bestPoly.empty()) {
        outConfidence = 0.0;
        return false;
    }

    std::vector<cv::Point2f> ordered = orderQuadRobust(orderQuadPoints(bestPoly));
    cv::cornerSubPix(gray, ordered, cv::Size(5, 5), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 25, 0.01));
    corners = ordered;

    double textureScore = std::clamp(stdVal / 60.0, 0.3, 1.0);
    double areaNorm = bestArea / frameArea;
    double areaScore = std::clamp(areaNorm / 0.15, 0.0, 1.0);
    double edgeScore = std::clamp(edgeRatio / 0.04, 0.2, 1.0);
    outConfidence = 0.55 + 0.15 * textureScore + 0.15 * areaScore + 0.15 * edgeScore;

    g_lastAdaptiveParams.blurK = blurK;
    g_lastAdaptiveParams.cannyLow = cannyLow;
    g_lastAdaptiveParams.cannyHigh = cannyHigh;
    g_lastAdaptiveParams.morphK = morphK;
    g_lastAdaptiveParams.meanGray = meanVal;
    g_lastAdaptiveParams.stdGray = stdVal;
    g_lastAdaptiveParams.confidenceBoost = outConfidence;

    return true;
}

// ============= TemporalValidator =============
TemporalValidator::TemporalValidator() 
    : is_initialized(false), consecutive_detections(0), frame_count(0) {}

void TemporalValidator::addDetection(const std::vector<cv::Point2f>& corners, double confidence) {
    detection_history.push_back(corners);
    confidence_history.push_back(confidence);
    if (detection_history.size() > (size_t)Config::detection.historySize) {
        detection_history.erase(detection_history.begin());
        confidence_history.erase(confidence_history.begin());
    }
    if (confidence > Config::detection.minConfidence) consecutive_detections++;
    else consecutive_detections = 0;
    frame_count++;
    if (!is_initialized && consecutive_detections >= Config::detection.minDetections) {
        is_initialized = true;
    }
    updatePrediction();
}

void TemporalValidator::updatePrediction() {
    if (detection_history.size() < 2) return;

    const auto& current = detection_history.back();
    const auto& previous = detection_history[detection_history.size() - 2];

    predicted_corners.clear();
    predicted_corners.reserve(4);

    for (size_t i = 0; i < current.size() && i < previous.size(); ++i) {
        cv::Point2f motion = current[i] - previous[i];
        predicted_corners.push_back(current[i] + motion * 0.5f);
    }
}

double TemporalValidator::getCurrentConfidence() const {
    if (confidence_history.empty()) return 0.0;

    double sum = 0.0;
    for (double conf : confidence_history) {
        sum += conf;
    }
    return sum / confidence_history.size();
}

bool TemporalValidator::isValid() const {
    return is_initialized && getCurrentConfidence() > Config::detection.minConfidence;
}

cv::Rect TemporalValidator::getAdaptiveROI(const cv::Size& img_size, double expansion_factor) const {
    if (detection_history.empty()) {
        return cv::Rect(0, 0, img_size.width, img_size.height);
    }

    const auto& corners = detection_history.back();
    if (corners.size() != 4) {
        return cv::Rect(0, 0, img_size.width, img_size.height);
    }

    float min_x = corners[0].x, max_x = corners[0].x;
    float min_y = corners[0].y, max_y = corners[0].y;

    for (const auto& corner : corners) {
        min_x = std::min(min_x, corner.x);
        max_x = std::max(max_x, corner.x);
        min_y = std::min(min_y, corner.y);
        max_y = std::max(max_y, corner.y);
    }

    float width = max_x - min_x;
    float height = max_y - min_y;
    float expand_w = width * (expansion_factor - 1.0) * 0.5;
    float expand_h = height * (expansion_factor - 1.0) * 0.5;

    int roi_x = std::max(0, static_cast<int>(min_x - expand_w));
    int roi_y = std::max(0, static_cast<int>(min_y - expand_h));
    int roi_w = std::min(img_size.width - roi_x, static_cast<int>(width + 2 * expand_w));
    int roi_h = std::min(img_size.height - roi_y, static_cast<int>(height + 2 * expand_h));

    return cv::Rect(roi_x, roi_y, roi_w, roi_h);
}

// ============= MultiScaleDetector =============
MultiScaleDetector::MultiScaleDetector() : scales(Config::detection.multiscaleScales) {}

std::pair<std::vector<cv::Point2f>, double> MultiScaleDetector::detectMultiScale(
    const cv::Mat& image, const cv::Rect& roi)
{
    std::vector<std::pair<std::vector<cv::Point2f>, double>> detections;
    cv::Mat roi_image = image(roi);
    for (double scale : scales) {
        cv::Mat scaled;
        if (std::abs(scale - 1.0) < 1e-6) {
            scaled = roi_image;
        } else {
            cv::resize(roi_image, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
        }
        std::vector<cv::Point2f> corners;
        double conf = 0.0;
        if (detectA4CornersAdaptive(scaled, corners, conf)) {
            for (auto& p : corners) {
                p.x = p.x / scale + roi.x;
                p.y = p.y / scale + roi.y;
            }
            detections.emplace_back(corners, conf);
        }
    }
    if (detections.empty()) return {std::vector<cv::Point2f>(), 0.0};
    std::sort(detections.begin(), detections.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    return detections.front();
}

bool detectA4CornersRobust(const cv::Mat& frame,
                           std::vector<cv::Point2f>& corners,
                           TemporalValidator& validator,
                           MultiScaleDetector& detector)
{
    corners.clear();
    if (frame.empty()) return false;

    cv::Rect roi = validator.getAdaptiveROI(frame.size(), Config::detection.adaptiveRoiExpansion);
    roi &= cv::Rect(0, 0, frame.cols, frame.rows);

    auto det = detector.detectMultiScale(frame, roi);
    if (det.first.empty()) {
        roi = cv::Rect(0, 0, frame.cols, frame.rows);
        det = detector.detectMultiScale(frame, roi);
    }

    if (!det.first.empty()) {
        if (det.first.size() == 4) {
            corners = det.first;
            validator.addDetection(corners, det.second);
            if (validator.isValid()) return true;
        }
    }

    if (validator.isValid() && !validator.predicted_corners.empty()) {
        corners = validator.predicted_corners;
        validator.addDetection(corners, 0.4);
        return true;
    }
    return false;
}
