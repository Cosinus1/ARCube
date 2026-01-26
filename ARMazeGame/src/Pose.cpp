#include "Pose.hpp"
#include <cmath>

bool poseFromQuadHomography(const std::vector<cv::Point2f>& quadTLBLBRTR,
                            const cv::Mat& K, const cv::Mat& D,
                            double side, cv::Mat& rvec, cv::Mat& tvec) {
    if (quadTLBLBRTR.size() != 4) return false;
    
    // Input order: TL, BL, BR, TR
    // Convert to standard order for getPerspectiveTransform: TL, TR, BR, BL
    std::vector<cv::Point2f> quadTLTRBRBL(4);
    quadTLTRBRBL[0] = quadTLBLBRTR[0];  // TL 
    quadTLTRBRBL[1] = quadTLBLBRTR[3];  // TR 
    quadTLTRBRBL[2] = quadTLBLBRTR[2];  // BR 
    quadTLTRBRBL[3] = quadTLBLBRTR[1];  // BL 
    
    const double s = side;
    
    // Object points (square on XY plane, Z=0)
    // Use standard right-handed coordinate system
    std::vector<cv::Point2f> obj2 = {
        { (float)(-s * 0.5f), (float)(+s * 0.5f) }, // TL: left, top
        { (float)(+s * 0.5f), (float)(+s * 0.5f) }, // TR: right, top
        { (float)(+s * 0.5f), (float)(-s * 0.5f) }, // BR: right, bottom
        { (float)(-s * 0.5f), (float)(-s * 0.5f) }  // BL: left, bottom
    };

    // Undistort image points
    std::vector<cv::Point2f> und;
    cv::undistortPoints(quadTLTRBRBL, und, K, D);

    // Compute homography from object to normalized image coordinates
    cv::Mat H = cv::getPerspectiveTransform(obj2, und);

    // Extract rotation and translation from homography
    cv::Mat h1 = H.col(0), h2 = H.col(1), h3 = H.col(2);
    double lambda = 1.0 / cv::norm(h1);
    
    cv::Mat r1 = lambda * h1;
    cv::Mat r2 = lambda * h2;
    cv::Mat r3 = r1.cross(r2);

    // Construct rotation matrix
    cv::Mat R(3, 3, CV_64F);
    r1.copyTo(R.col(0));
    r2.copyTo(R.col(1));
    r3.copyTo(R.col(2));
    
    // Enforce orthogonality using SVD
    cv::SVD svd(R);
    R = svd.u * svd.vt;
    
    if (cv::determinant(R) < 0) {
        R = -R;
    }

    // Translation
    cv::Mat t = lambda * h3;

    // Ensure the object is in front of the camera (positive Z in camera coordinates)
    if (t.at<double>(2) < 0) {
        R = -R;
        t = -t;
    }

    cv::Rodrigues(R, rvec);
    tvec = t.clone();
    
    return true;
}

void projectSquareBase(double side, const cv::Mat& K, const cv::Mat& D,
                       const cv::Mat& rvec, const cv::Mat& tvec,
                       std::vector<cv::Point2f>& out2d) {
    float a = 0.5f * (float)side;
    std::vector<cv::Point3f> base = {
        {-a, +a, 0},  // TL
        {+a, +a, 0},  // TR
        {+a, -a, 0},  // BR
        {-a, -a, 0}   // BL
    };
    cv::projectPoints(base, rvec, tvec, K, D, out2d);
}

double computeReprojectionError(const std::vector<cv::Point2f>& imagePoints,
                                const std::vector<cv::Point2f>& projectedPoints) {
    if (imagePoints.size() != projectedPoints.size() || imagePoints.empty()) {
        return std::numeric_limits<double>::max();
    }
    
    double totalError = 0.0;
    for (size_t i = 0; i < imagePoints.size(); ++i) {
        double dx = imagePoints[i].x - projectedPoints[i].x;
        double dy = imagePoints[i].y - projectedPoints[i].y;
        totalError += std::sqrt(dx * dx + dy * dy);
    }
    
    return totalError / imagePoints.size();
}