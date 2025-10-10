// AR - 2025-2026
// S. Mavromatis
// Détections : contours, approximation par un polygone à 4 sommets

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Charger l'image
    cv::Mat img = cv::imread("data/image_1.jpg");
    if (img.empty()) {
        std::cerr << "Erreur: impossible de charger l'image !" << std::endl;
        return -1;
    }

    cv::Mat gray, thresh;
    // Conversion en niveaux de gris
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Flou "léger" pour réduire le bruit
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    // Seuillage
    cv::threshold(gray, thresh, 200, 255, cv::THRESH_BINARY);

    // Recherche des contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        std::cerr << "Aucun contour détecté !" << std::endl;
        return -1;
    }

    // Chercher le plus grand contour (aire)
    double maxArea = 0;
    int maxIdx = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIdx = i;
        }
    }

    // Approximation du contour par un polygone
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contours[maxIdx], approx, 10, true);

    std::cout << "Sommets : " << std::endl;
    for (size_t i = 0; i < approx.size(); i++) {
        std::cout << "(" << approx[i].x << ", " << approx[i].y << ")" << std::endl;
        // Dessin des points sur l'image
        cv::circle(img, approx[i], 10, cv::Scalar(0, 0, 255), -1);
    }

    // Affichage
    cv::drawContours(img, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Sommets : ", img);
    cv::waitKey(0);

    return 0;
}
