#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <cstdio>
#include <cmath>
#include <cstdint>

#include "Utils.hpp"
#include "Detection.hpp"
#include "Drawing.hpp"
#include "Quat.hpp"
#include "Pose.hpp"
#include "Calibration.hpp"

int main(int argc, char** argv){
    if (argc < 2){
        std::cerr <<
        "Usage:\n"
        "  ./AROne image  <image.jpg> [camera.yaml]\n"
        "  ./AROne video  <video.mp4> [camera.yaml]\n"
        "  ./AROne webcam [camIndex]  [camera.yaml]\n";
        return 1;
    }

    std::string mode = argv[1];
    for (auto& ch:mode) ch = (char)std::tolower((unsigned char)ch);

    bool useImage=false, useVideo=false, useWebcam=false;
    if      (mode=="image")  useImage=true;
    else if (mode=="video")  useVideo=true;
    else if (mode=="webcam") useWebcam=true;
    else {
        std::cerr<<"Mode inconnu: "<<mode<<"\n";
        return 1;
    }

    std::filesystem::path exePath = std::filesystem::path(argv[0]);
    std::filesystem::path DATA_ROOT  = resolveDataRoot(exePath);
    std::filesystem::path IMAGE_DIR  = DATA_ROOT / "image";
    std::filesystem::path VIDEO_DIR  = DATA_ROOT / "video";
    std::filesystem::path CAMERA_DIR = DATA_ROOT / "yaml";

    std::filesystem::path mediaPath;
    int camIndex = 0;
    std::string yamlName;
    std::filesystem::path yamlPath;

    if (useImage){
        if (argc < 3){ std::cerr<<"Argument media manquant\n"; return 1; }
        std::filesystem::path mediaName = argv[2];
        mediaPath = mediaName.is_absolute() ? mediaName : (IMAGE_DIR / mediaName);
        if (!std::filesystem::exists(mediaPath)) {
            std::cerr << "Erreur : fichier image introuvable : " << mediaPath.string() << "\n";
            return 1;
        }
        yamlName = (argc >= 4) ? argv[3] : "image.yaml";
    }
    else if (useVideo){
        if (argc < 3){ std::cerr<<"Argument media manquant\n"; return 1; }
        std::filesystem::path mediaName = argv[2];
        mediaPath = mediaName.is_absolute() ? mediaName : (VIDEO_DIR / mediaName);
        if (!std::filesystem::exists(mediaPath)) {
            std::cerr << "Erreur : fichier video introuvable : " << mediaPath.string() << "\n";
            return 1;
        }
        yamlName = (argc >= 4) ? argv[3] : "video.yaml";
    }
    else {
        if (argc >= 3) camIndex = std::atoi(argv[2]);
        if (argc >= 4) {
            yamlName = argv[3];
        } else {
            yamlName.clear();
        }
    }

    yamlPath = std::filesystem::path(yamlName).is_absolute()
            ? std::filesystem::path(yamlName)
            : (CAMERA_DIR / yamlName);

    cv::Mat K, D;
    cv::Size calibSize;

    if (useWebcam){
        if (yamlName.empty()) {
            std::cout << "Aucun fichier YAML fourni, lancement de la calibration live...\n";
            if (!runLiveCalibration(camIndex, (CAMERA_DIR / "temp.yaml").string(), calibSize, K, D)) {
                std::cerr << "Calibration annulee ou echouee.\n";
                return 1;
            }

            cv::destroyAllWindows();
            cv::waitKey(1);

            std::cout << "\nEntrez un nom pour sauvegarder cette calibration : " << std::flush;
            std::string userName;
            std::getline(std::cin, userName);

            if (userName.empty()) {
                auto t = std::time(nullptr);
                std::tm tm{};
            #ifdef _WIN32
                localtime_s(&tm, &t);
            #else
                localtime_r(&t, &tm);
            #endif
                char buf[64];
                std::strftime(buf, sizeof(buf), "webcam_%Y-%m-%d_%H-%M", &tm);
                userName = buf;
            }

            std::filesystem::create_directories(CAMERA_DIR);
            std::filesystem::path finalPath = CAMERA_DIR / (userName + ".yaml");

            if (std::filesystem::exists(finalPath)) {
                std::cout << "Le fichier " << finalPath << " existe deja. Ecraser ? [y/N] ";
                std::string ans;
                std::getline(std::cin, ans);
                if (ans != "y" && ans != "Y") {
                    std::cout << "Annulation de la sauvegarde.\n";
                } else {
                    std::filesystem::rename(CAMERA_DIR / "temp.yaml", finalPath);
                    std::cout << "Calibration sauvegardee sous : " << finalPath << "\n";
                }
            } else {
                std::filesystem::rename(CAMERA_DIR / "temp.yaml", finalPath);
                std::cout << "Calibration sauvegardee sous : " << finalPath << "\n";
            }

            yamlPath = finalPath;
            std::cout << "Calibration terminee, passage automatique en AR live.\n";
        }
        else {
            yamlPath = std::filesystem::path(yamlName).is_absolute()
                    ? std::filesystem::path(yamlName)
                    : (CAMERA_DIR / yamlName);

            if (!loadCalibration(yamlPath.string(), K, D, calibSize))
                return 1;
        }
    } else {
        if (!std::filesystem::exists(yamlPath)) {
            std::cerr << "Erreur : fichier YAML introuvable : " << yamlPath.string() << "\n";
            return 1;
        }
        if (!loadCalibration(yamlPath.string(), K, D, calibSize)) return 1;
    }

    const double model_size       = 0.50;
    const double poseAlpha        = 0.45;
    const float  qAlpha           = 0.60f;
    const double cubeBaseScale    = 0.25;
    const double cubeHeightScale  = 0.25;
    const float  half             = 0.5f * (float)model_size * (float)cubeBaseScale;
    const float  h                = (float)model_size * (float)cubeHeightScale;

    auto make_process = [&](const cv::Mat& K_eff){
        return [&](cv::Mat& frame){
            cv::Mat gray; cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> quad; static bool hadSmoothQuad=false;
            static std::array<cv::Point2f,4> smoothQ, quadPrev;
            bool found = findLargestQuad(gray, quad);
            if (!found && !hadSmoothQuad) return;
            if (!found) quad = std::vector<cv::Point2f>(quadPrev.begin(), quadPrev.end());

            std::array<cv::Point2f,4> quadSm;
            if (!hadSmoothQuad){ for(int i=0;i<4;i++){ smoothQ[i]=quad[i]; quadSm[i]=quad[i]; } hadSmoothQuad=true; }
            else { for(int i=0;i<4;i++){ smoothQ[i]= qAlpha*smoothQ[i] + (1.0f-qAlpha)*quad[i]; quadSm[i]=smoothQ[i]; } }
            quadPrev = quadSm;

            drawLabeledQuad(frame, {quadSm[0],quadSm[1],quadSm[2],quadSm[3]});

            cv::Mat r_D, t_D;
            bool okD = poseFromQuadHomography({quadSm[0],quadSm[1],quadSm[2],quadSm[3]}, K_eff, D, model_size, r_D, t_D);

            cv::Mat D_zero = cv::Mat::zeros(1,5,CV_64F);
            cv::Mat r_Z, t_Z;
            bool okZ = poseFromQuadHomography({quadSm[0],quadSm[1],quadSm[2],quadSm[3]}, K_eff, D_zero, model_size, r_Z, t_Z);

            if (!okD && !okZ) return;

            auto reprojErr = [&](const cv::Mat& r, const cv::Mat& t, const cv::Mat& K, const cv::Mat& Dcoefs){
                std::vector<cv::Point2f> baseProj;
                projectSquareBase(model_size, K, Dcoefs, r, t, baseProj);
                double e=0; for(int i=0;i<4;i++) e += cv::norm(baseProj[i] - quadSm[i]);
                return e/4.0;
            };

            double eD = okD ? reprojErr(r_D,t_D,K_eff,D)        : 1e9;
            double eZ = okZ ? reprojErr(r_Z,t_Z,K_eff,D_zero)   : 1e9;

            static bool lastUsedZero = false;
            bool useZero = (eZ + 0.2 < eD);
            if (lastUsedZero && eZ < eD + 0.5) useZero = true;

            cv::Mat rbest = useZero ? r_Z : r_D;
            cv::Mat tbest = useZero ? t_Z : t_D;
            cv::Mat D_use = useZero ? D_zero : D;
            lastUsedZero = useZero;

            static bool hasPrevPose=false; static cv::Mat rvecPrev, tvecPrev;
            cv::Mat rdisp, tdisp;
            if (hasPrevPose){
                Quat qPrev=rotvecToQuat(rvecPrev), qCurr=rotvecToQuat(rbest);
                Quat qMix = quatSlerp(qPrev, qCurr, poseAlpha);
                rdisp = quatToRotvec(qMix);
                tdisp = tvecPrev*(1.0-poseAlpha) + tbest*poseAlpha;
            } else { rdisp=rbest.clone(); tdisp=tbest.clone(); hasPrevPose=true; }

            std::vector<cv::Point2f> baseProj; projectSquareBase(model_size, K_eff, D_use, rdisp, tdisp, baseProj);
            double meanRes=0.0; for (int i=0;i<4;i++) meanRes+=cv::norm(baseProj[i]-quadSm[i]); meanRes/=4.0;
            if (meanRes > 3.5) { rdisp=rbest.clone(); tdisp=tbest.clone(); }

            rvecPrev=rdisp.clone(); tvecPrev=tdisp.clone();

            std::vector<cv::Point3f> cubePts = {
                {-half,-half,0.f},{+half,-half,0.f},{+half,+half,0.f},{-half,+half,0.f},
                {-half,-half,h  },{+half,-half,h  },{+half,+half,h  },{-half,+half,h  }
            };

            std::vector<cv::Point2f> cube2D; cv::projectPoints(cubePts, rdisp, tdisp, K_eff, D_use, cube2D);
            drawCubeFaces(frame, cubePts, rdisp, tdisp, K_eff, D_use);
            drawAxes(frame, K_eff, D_use, rdisp, tdisp, (float)(0.4*model_size));
        };
    };

    if (useWebcam){
        cv::VideoCapture cap(camIndex);
        if (!cap.isOpened()){
            std::cerr<<"Impossible d'ouvrir la webcam index "<<camIndex<<"\n"; return 1;
        }

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()){ std::cerr<<"Webcam: frame vide\n"; return 1; }

        Rot rot = decideRotationFromSizes(calibSize, frame.size());
        if (rot != Rot::NONE) rotateFrameInPlace(frame, rot);

        cv::Mat K_eff = scaleIntrinsics(K, calibSize, frame.size());

        auto process = make_process(K_eff);

        process(frame);
        cv::imshow("Plan + Repere 3D", frame);

        while(true){
            if (!cap.read(frame) || frame.empty()) break;
            if (rot != Rot::NONE) rotateFrameInPlace(frame, rot);
            process(frame);
            cv::imshow("Plan + Repere 3D", frame);
            int k = cv::waitKey(1);
            if (k==27 || k=='q') break;
            if (k=='r'){
                rot = (rot==Rot::CCW ? Rot::CW : (rot==Rot::CW ? Rot::NONE : Rot::CCW));
            }
        }
        return 0;
    }

    if (useImage){
        cv::Mat img = cv::imread(mediaPath.string());
        if (img.empty()){ std::cerr<<"Impossible de charger "<<mediaPath<<"\n"; return 1; }

        Rot rot = decideRotationFromSizes(calibSize, img.size());
        if (rot != Rot::NONE)
            rotateFrameInPlace(img, rot);

        cv::Mat K_eff = scaleIntrinsics(K, calibSize, img.size());

        auto process = make_process(K_eff);
        process(img);
        cv::resize(img, img, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        cv::imshow("Plan + Repere 3D", img);
        cv::waitKey(0);
        return 0;
    }

    cv::VideoCapture cap(mediaPath.string());
    if (!cap.isOpened()){ std::cerr<<"Impossible d'ouvrir "<<mediaPath<<"\n"; return 1; }

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()){ std::cerr<<"Video vide\n"; return 1; }

    Rot rot = decideRotationFromSizes(calibSize, frame.size());
    if (rot != Rot::NONE)
        rotateFrameInPlace(frame, rot);

    cv::Mat K_eff = scaleIntrinsics(K, calibSize, frame.size());

    auto process = make_process(K_eff);
    process(frame);
    cv::imshow("Plan + Repere 3D", frame);

    while(true){
        if (!cap.read(frame) || frame.empty()) break;
        if (rot != Rot::NONE)
            rotateFrameInPlace(frame, rot);

        process(frame);
        cv::imshow("Plan + Repere 3D", frame);
        int k=cv::waitKey(20);
        if (k==27 || k=='q') break;
    }
    return 0;
}
