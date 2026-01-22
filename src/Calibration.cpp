#include "Calibration.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

static const cv::Size kBoardSize(9,6);
static const float    kSquareSize = 0.024f;

bool runLiveCalibration(int camIndex,
                        const std::string& yamlPath,
                        cv::Size& calibSize_out,
                        cv::Mat& K_out, cv::Mat& D_out)
{
    cv::VideoCapture cap(camIndex);
    if (!cap.isOpened()){
        std::cerr << "Calibration: impossible d'ouvrir la webcam index " << camIndex << "\n";
        return false;
    }

    std::vector<std::vector<cv::Point2f>> imgPoints;
    std::vector<std::vector<cv::Point3f>> objPoints;

    std::vector<cv::Point3f> obj;
    obj.reserve(kBoardSize.area());
    for (int j=0; j<kBoardSize.height; ++j)
        for (int i=0; i<kBoardSize.width; ++i)
            obj.emplace_back(i*kSquareSize, j*kSquareSize, 0.f);

    std::cout
        << "[Calibration]\n"
        << " - Montre la mire (checkerboard " << kBoardSize.width << "x" << kBoardSize.height << ")\n"
        << " - Appuie sur 'c' pour CAPTURER une vue quand la mire est bien detectee\n"
        << " - Appuie sur 'q' pour ABANDONNER\n";

    const int needed = 30;
    cv::Mat frame, gray;

    auto areaOf = [](const std::vector<cv::Point2f>& c){
        std::vector<cv::Point> poly = { (cv::Point)c.front(), (cv::Point)c.back() };
        cv::Rect bb = cv::boundingRect(c);
        return (double)bb.area();
    };
    auto rmsShift = [](const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b){
        if (a.size()!=b.size() || a.empty()) return 1e9;
        double s=0; for(size_t i=0;i<a.size();++i) s+=cv::norm(a[i]-b[i])*cv::norm(a[i]-b[i]);
        return std::sqrt(s/a.size());
    };
    bool autoMode = true;
    std::vector<cv::Point2f> lastCaptured;
    double lastArea = 0.0;
    int64_t lastTick = cv::getTickCount();
    const double tick2ms = 1000.0 / cv::getTickFrequency();

    while ((int)imgPoints.size() < needed){
        if (!cap.read(frame) || frame.empty()) { std::cerr<<"Calibration: frame vide\n"; break; }
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(
            gray, kBoardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
        );

        if (found){
            cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1),
            cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::MAX_ITER, 30, 0.01));
            cv::drawChessboardCorners(frame, kBoardSize, corners, true);
            cv::putText(frame, autoMode? "Mire detectee - AUTO capture" : "Mire detectee - 'c' pour capturer", {20,30}, 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,0,0}, 2);

            if (autoMode){
                int64_t now = cv::getTickCount();
                double dt_ms = (now - lastTick) * tick2ms;

                double curArea = areaOf(corners);
                double areaRelChange = (lastArea>0.0)? std::abs(curArea - lastArea) / lastArea : 1.0;
                double move = lastCaptured.empty()? 1e9 : rmsShift(corners, lastCaptured);

                bool enoughChange = (move > 15.0) || (areaRelChange > 0.12);
                bool enoughTime   = (dt_ms > 700.0);

                if (enoughTime && enoughChange){
                    imgPoints.push_back(corners);
                    std::vector<cv::Point3f> obj; obj.reserve(kBoardSize.area());
                    for (int j=0; j<kBoardSize.height; ++j)
                        for (int i=0; i<kBoardSize.width; ++i)
                            obj.emplace_back(i*kSquareSize, j*kSquareSize, 0.f);
                    objPoints.push_back(obj);

                    lastCaptured = corners;
                    lastArea = curArea;
                    lastTick = now;
                }
            }
        } else {
            cv::putText(frame, "Cherche la mire...", {20,30},
            cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 2);
        }

        std::ostringstream ss; 
        ss << "Vues: " << imgPoints.size() << "/" << needed 
        << "   [a] Auto:" << (autoMode? "ON":"OFF")
        << "   [c] Capture   [q] Quit";
        cv::putText(frame, ss.str(), {20,60}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 2);

        cv::imshow("Calibration", frame);
        int k = cv::waitKey(1);
        if (k=='q' || k==27) { cv::destroyWindow("Calibration"); return false; }
        if (k=='a') autoMode = !autoMode;
        if (k=='c' && found){
            imgPoints.push_back(corners);
            std::vector<cv::Point3f> obj; obj.reserve(kBoardSize.area());
            for (int j=0; j<kBoardSize.height; ++j)
                for (int i=0; i<kBoardSize.width; ++i)
                    obj.emplace_back(i*kSquareSize, j*kSquareSize, 0.f);
            objPoints.push_back(obj);
            lastCaptured = corners;
            lastArea = areaOf(corners);
            lastTick = cv::getTickCount();
        }
    }
    cv::destroyWindow("Calibration");

    if ((int)imgPoints.size() < needed){
        std::cerr<<"Calibration: pas assez d'images capturees\n";
        return false;
    }

    // calibrate
    cv::Mat K, D;
    std::vector<cv::Mat> rvecs, tvecs;
    calibSize_out = frame.size(); 
    double rms = cv::calibrateCamera(objPoints, imgPoints, calibSize_out, K, D, rvecs, tvecs,
                                     cv::CALIB_RATIONAL_MODEL); 
    std::cout << "Calibration OK. RMS = " << rms << "\n";

    // save
    std::filesystem::create_directories(std::filesystem::path(yamlPath).parent_path());
    cv::FileStorage fs(yamlPath, cv::FileStorage::WRITE);
    if (!fs.isOpened()){
        std::cerr << "Calibration: impossible d'ecrire le YAML: " << yamlPath << "\n";
        return false;
    }
    fs << "image_width"  << calibSize_out.width;
    fs << "image_height" << calibSize_out.height;
    fs << "camera_matrix" << K;
    fs << "distortion_coefficients" << D;
    fs.release();

    std::cout << "YAML ecrit : " << yamlPath << "\n";

    K.convertTo(K_out, CV_64F);
    D.convertTo(D_out, CV_64F);
    
    cv::putText(frame, "Calibration OK ! Retirez la mire.", {20,40},
    cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,0,0}, 2);
    cv::imshow("Calibration", frame);
    cv::waitKey(700);

    return true;
}
