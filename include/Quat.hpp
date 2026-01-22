#pragma once
#include <opencv2/opencv.hpp>

struct Quat { double w,x,y,z; };

Quat matToQuat(const cv::Matx33d& R);
cv::Matx33d quatToMat(const Quat& q);
Quat quatNorm(const Quat& a);
Quat quatSlerp(const Quat& qa, const Quat& qb, double t);
Quat rotvecToQuat(const cv::Mat& rvec);
cv::Mat quatToRotvec(const Quat& q);
