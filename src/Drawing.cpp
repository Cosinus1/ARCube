#include "Drawing.hpp"

void drawLabeledQuad(cv::Mat& frame, const std::vector<cv::Point2f>& q){
    cv::Scalar col = {0,255,255};
    for (int i=0;i<4;++i) cv::line(frame, q[i], q[(i+1)%4], col, 2, cv::LINE_AA);
    for (int i=0;i<4;++i){
        cv::circle(frame, q[i], 5, {0,0,255}, -1, cv::LINE_AA);
        std::ostringstream ss; ss<<"("<<(int)q[i].x<<","<<(int)q[i].y<<")";
        cv::putText(frame, ss.str(), q[i]+cv::Point2f(6,-6), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 2);
    }
}

void drawAxes(cv::Mat& frame, const cv::Mat& K, const cv::Mat& D,
              const cv::Mat& rvec, const cv::Mat& tvec, float L){
    std::vector<cv::Point3f> obj = { {0,0,0}, {L,0,0}, {0,L,0}, {0,0,L} };
    std::vector<cv::Point2f> img;
    cv::projectPoints(obj, rvec, tvec, K, D, img);
    cv::line(frame, img[0], img[1], {0,0,255}, 2, cv::LINE_AA); // X
    cv::line(frame, img[0], img[2], {0,255,0}, 2, cv::LINE_AA); // Y
    cv::line(frame, img[0], img[3], {255,0,0}, 2, cv::LINE_AA); // Z
    cv::putText(frame, "X", img[1]+cv::Point2f(5,-5), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,255}, 2);
    cv::putText(frame, "Y", img[2]+cv::Point2f(5,-5), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);
    cv::putText(frame, "Z", img[3]+cv::Point2f(5,-5), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,0,0}, 2);
}

void drawCubeWireframe(cv::Mat& frame, const std::vector<cv::Point2f>& imgPts){
    auto line = [&](int i, int j){ cv::line(frame, imgPts[i], imgPts[j], {0,255,255}, 2, cv::LINE_AA); };
    line(0,1); line(1,2); line(2,3); line(3,0);
    line(4,5); line(5,6); line(6,7); line(7,4);
    line(0,4); line(1,5); line(2,6); line(3,7);
}

void drawCubeFaces(cv::Mat& frame,
                   const std::vector<cv::Point3f>& cubePts3D,
                   const cv::Mat& rvec, const cv::Mat& tvec,
                   const cv::Mat& K, const cv::Mat& D){
    std::vector<cv::Point2f> pts2D;
    cv::projectPoints(cubePts3D, rvec, tvec, K, D, pts2D);

    cv::Matx33d R; cv::Rodrigues(rvec, R);
    cv::Vec3d tv(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    std::vector<cv::Vec3d> camPts(8);
    for (int i=0;i<8;++i){
        cv::Vec3d X(cubePts3D[i].x, cubePts3D[i].y, cubePts3D[i].z);
        camPts[i] = R * X + tv;
    }
    const int F[6][4] = { {0,1,2,3},{4,5,6,7},{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7} };
    struct Face { int idx; double zavg; };
    std::vector<Face> vis;
    for (int f=0; f<6; ++f){
        cv::Vec3d a=camPts[F[f][0]], b=camPts[F[f][1]], c=camPts[F[f][2]];
        cv::Vec3d v1=b-a, v2=c-b;
        cv::Vec3d N(v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]);
        double z=0; for(int k=0;k<4;++k) z+=camPts[F[f][k]][2]; z/=4.0;
        if (N[2] < 0.0) vis.push_back({f,z});
    }
    std::sort(vis.begin(), vis.end(), [](const Face&a,const Face&b){ return a.zavg>b.zavg; });

    const cv::Scalar cols[6]={{60,60,200},{60,200,60},{200,60,60},{200,200,60},{200,60,200},{60,200,200}};
    const double opacite=0.85;

    for (auto& f: vis){
        std::vector<cv::Point> poly; poly.reserve(4);
        for (int k=0;k<4;++k) poly.emplace_back(pts2D[F[f.idx][k]]);
        cv::Mat overlay=frame.clone();
        cv::fillConvexPoly(overlay, poly, cols[f.idx], cv::LINE_AA);
        cv::addWeighted(overlay, opacite, frame, 1.0-opacite, 0.0, frame);
        cv::polylines(frame, poly, true, cv::Scalar(20,20,20), 1, cv::LINE_AA);
    }
}
