#include "Pose.hpp"

bool poseFromQuadHomography(const std::vector<cv::Point2f>& quadTLTRBRBL,
                            const cv::Mat& K, const cv::Mat& D,
                            double side, cv::Mat& rvec, cv::Mat& tvec)
{
    const double s=side;
    std::vector<cv::Point2f> obj2 = {
        { (float)-s*0.5f, (float)-s*0.5f }, // TL
        { (float)+s*0.5f, (float)-s*0.5f }, // TR
        { (float)+s*0.5f, (float)+s*0.5f }, // BR
        { (float)-s*0.5f, (float)+s*0.5f }  // BL
    };

    std::vector<cv::Point2f> und;
    cv::undistortPoints(quadTLTRBRBL, und, K, D); // (x/z, y/z)

    cv::Mat H = cv::getPerspectiveTransform(obj2, und); // 3x3

    cv::Mat h1=H.col(0), h2=H.col(1), h3=H.col(2);
    double lambda = 1.0 / cv::norm(h1);
    cv::Mat r1 = lambda*h1, r2 = lambda*h2, r3 = r1.cross(r2);

    cv::Mat R(3,3,CV_64F); r1.copyTo(R.col(0)); r2.copyTo(R.col(1)); r3.copyTo(R.col(2));
    cv::SVD svd(R); R = svd.u * svd.vt;
    if (cv::determinant(R) < 0) R = -R;

    cv::Mat t = lambda*h3;

    if (R.at<double>(2,2) > 0){ R = -R; t = -t; }

    cv::Rodrigues(R, rvec);
    tvec = t.clone();
    return true;
}

void projectSquareBase(double side, const cv::Mat& K, const cv::Mat& D,
                       const cv::Mat& rvec, const cv::Mat& tvec,
                       std::vector<cv::Point2f>& out2d)
{
    float a=0.5f*(float)side;
    std::vector<cv::Point3f> base={{-a,-a,0},{+a,-a,0},{+a,+a,0},{-a,+a,0}};
    cv::projectPoints(base, rvec, tvec, K, D, out2d);
}
