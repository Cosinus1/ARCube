#include "Detection.hpp"
#include <algorithm>

std::vector<cv::Point2f> orderQuadRobust(const std::vector<cv::Point2f>& quadRaw){
    std::vector<cv::Point2f> P = quadRaw;
    cv::Point2f c(0.f,0.f); for (auto &p:P) c += p; c *= 0.25f;
    std::sort(P.begin(), P.end(), [&](const cv::Point2f&a, const cv::Point2f&b){
        float aa = std::atan2(a.y - c.y, a.x - c.x);
        float bb = std::atan2(b.y - c.y, b.x - c.x);
        return aa < bb;
    });
    auto signedArea = [&](const std::vector<cv::Point2f>& Q){
        double A=0; for(int i=0;i<4;++i){ auto& u=Q[i]; auto& v=Q[(i+1)%4]; A += u.x*v.y - v.x*u.y; }
        return A;
    };
    if (signedArea(P) > 0.0) std::reverse(P.begin()+1, P.end());
    int tl=0; float best=P[0].x+P[0].y;
    for(int i=1;i<4;++i){ float s=P[i].x+P[i].y; if(s<best){ best=s; tl=i; } }
    std::vector<cv::Point2f> O(4); for(int i=0;i<4;++i) O[i]=P[(tl+i)%4];
    return O;
}

bool findLargestQuad(const cv::Mat& gray, std::vector<cv::Point2f>& quadOrdered){
    cv::Mat blur, edges; cv::GaussianBlur(gray, blur, cv::Size(3,3), 0);
    cv::Canny(blur, edges, 75, 200);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double maxArea=0.0; std::vector<cv::Point> best;
    for (auto& c : contours){
        if (c.size()<4) continue;
        double peri = cv::arcLength(c, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.008*peri, true);
        if (approx.size()==4 && cv::isContourConvex(approx)){
            double area = std::fabs(cv::contourArea(approx));
            if (area > maxArea){ maxArea=area; best=approx; }
        }
    }
    if (best.empty()) return false;

    cv::Mat ptsf(4,1,CV_32FC2);
    for (int i=0;i<4;++i) ptsf.at<cv::Point2f>(i,0)=best[i];

    cv::Mat gray8; if (gray.type()!=CV_8U) gray.convertTo(gray8, CV_8U); else gray8=gray;
    cv::cornerSubPix(gray8, ptsf, cv::Size(3,3), cv::Size(-1,-1),
                     cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::MAX_ITER, 40, 0.001));

    std::vector<cv::Point2f> q4(4);
    for (int i=0;i<4;++i) q4[i]=ptsf.at<cv::Point2f>(i,0);
    quadOrdered = orderQuadRobust(q4);
    return true;
}
