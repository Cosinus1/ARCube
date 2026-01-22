#include "Utils.hpp"
#include <cstdlib>

std::filesystem::path resolveDataRoot(const std::filesystem::path& exePath)
{
    if (const char* env = std::getenv("AR_DATA_DIR")) {
        std::filesystem::path p(env);
        if (std::filesystem::exists(p)) return std::filesystem::canonical(p);
    }

    std::filesystem::path exeDir = exePath.parent_path();

    std::vector<std::filesystem::path> candidates = {
        exeDir / ".." / ".." / "data",
        exeDir / ".." / "data",
        std::filesystem::current_path() / "data",
        std::filesystem::current_path() / ".." / "data"
    };

    for (auto& c : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(c, ec)) {
            return std::filesystem::canonical(c, ec);
        }
    }

    return std::filesystem::current_path();
}

bool loadCalibration(const std::string& yamlPath, cv::Mat& K, cv::Mat& D, cv::Size& calibSize){
    cv::FileStorage fs(yamlPath, cv::FileStorage::READ);
    if(!fs.isOpened()){
        std::cerr << "Erreur: impossible d'ouvrir le YAML: " << yamlPath << "\n";
        return false;
    }
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> D;
    int w=0,h=0;
    if (fs["image_width"].isInt())  w = (int)fs["image_width"];
    if (fs["image_height"].isInt()) h = (int)fs["image_height"];
    calibSize = (w>0 && h>0) ? cv::Size(w,h) : cv::Size();

    if (K.empty()){ std::cerr<<"Erreur: camera_matrix absente dans "<<yamlPath<<"\n"; return false; }
    if (D.empty()) D = cv::Mat::zeros(1,5,CV_64F);
    K.convertTo(K, CV_64F);
    D.convertTo(D, CV_64F);
    return true;
}

cv::Mat scaleIntrinsics(const cv::Mat& K0, cv::Size from, cv::Size to){
    if (from.width<=0 || from.height<=0 || from==to) return K0.clone();
    double sx = (double)to.width  / (double)from.width;
    double sy = (double)to.height / (double)from.height;
    cv::Mat K1 = K0.clone();
    K1.at<double>(0,0) *= sx; // fx
    K1.at<double>(1,1) *= sy; // fy
    K1.at<double>(0,2) *= sx; // cx
    K1.at<double>(1,2) *= sy; // cy
    return K1;
}

void rotateFrameInPlace(cv::Mat& img, Rot r){
    if (r==Rot::CW)  cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
    if (r==Rot::CCW) cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
}

Rot decideRotationFromSizes(const cv::Size& calibSize, const cv::Size& frameSize){
    bool calibPortrait = calibSize.height > calibSize.width;
    bool framePortrait = frameSize.height > frameSize.width;

    if (calibPortrait == framePortrait)
        return Rot::NONE;
    return Rot::CCW;
}
