#include "Quat.hpp"
#include <cmath>

Quat matToQuat(const cv::Matx33d& R){
    Quat q; double tr=R(0,0)+R(1,1)+R(2,2);
    if (tr>0){ double s=std::sqrt(tr+1.0)*2.0; q.w=0.25*s; q.x=(R(2,1)-R(1,2))/s; q.y=(R(0,2)-R(2,0))/s; q.z=(R(1,0)-R(0,1))/s; }
    else if (R(0,0)>R(1,1) && R(0,0)>R(2,2)){ double s=std::sqrt(1.0+R(0,0)-R(1,1)-R(2,2))*2.0; q.w=(R(2,1)-R(1,2))/s; q.x=0.25*s; q.y=(R(0,1)+R(1,0))/s; q.z=(R(0,2)+R(2,0))/s; }
    else if (R(1,1)>R(2,2)){ double s=std::sqrt(1.0+R(1,1)-R(0,0)-R(2,2))*2.0; q.w=(R(0,2)-R(2,0))/s; q.x=(R(0,1)+R(1,0))/s; q.y=0.25*s; q.z=(R(1,2)+R(2,1))/s; }
    else { double s=std::sqrt(1.0+R(2,2)-R(0,0)-R(1,1))*2.0; q.w=(R(1,0)-R(0,1))/s; q.x=(R(0,2)+R(2,0))/s; q.y=(R(1,2)+R(2,1))/s; q.z=0.25*s; }
    return q;
}

cv::Matx33d quatToMat(const Quat& q){
    double w=q.w,x=q.x,y=q.y,z=q.z;
    return cv::Matx33d(
        1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w,
        2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w,
        2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y
    );
}

Quat quatNorm(const Quat& a){
    double n=std::sqrt(a.w*a.w+a.x*a.x+a.y*a.y+a.z*a.z);
    return {a.w/n,a.x/n,a.y/n,a.z/n};
}

Quat quatSlerp(const Quat& qa, const Quat& qb, double t){
    Quat a=quatNorm(qa), b=quatNorm(qb);
    double dot=a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z;
    if (dot<0){ dot=-dot; b.w=-b.w; b.x=-b.x; b.y=-b.y; b.z=-b.z; }
    const double TH=0.9995;
    if (dot>TH){ Quat r{a.w+t*(b.w-a.w),a.x+t*(b.x-a.x),a.y+t*(b.y-a.y),a.z+t*(b.z-a.z)}; return quatNorm(r); }
    double th0=std::acos(dot), th=th0*t;
    double s0=std::cos(th)-dot*std::sin(th)/std::sin(th0);
    double s1=std::sin(th)/std::sin(th0);
    return quatNorm({a.w*s0+b.w*s1, a.x*s0+b.x*s1, a.y*s0+b.y*s1, a.z*s0+b.z*s1});
}

Quat rotvecToQuat(const cv::Mat& rvec){ cv::Matx33d R; cv::Rodrigues(rvec, R); return matToQuat(R); }

cv::Mat quatToRotvec(const Quat& q){ cv::Matx33d R=quatToMat(q); cv::Mat r; cv::Rodrigues(R, r); return r; }
