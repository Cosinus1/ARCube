#include "GUI.hpp"
#include <sstream>
#include <algorithm>

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    GUIManager* gui = static_cast<GUIManager*>(userdata);
    if (gui) {
        gui->handleMouse(event, x, y);
    }
}

GUIManager::GUIManager(int width, int height)
    : windowWidth(width)
    , windowHeight(height)
    , currentMode(AppMode::MODE_SELECT)
    , selectedWebcam(-1)
    , selectedMaze(MazeType::SIMPLE)
    , selectionComplete(false)
    , scrollOffset(0)
{
    createModeSelectButtons();
}

void GUIManager::render(cv::Mat& frame) {
    // Create frame if needed
    if (frame.empty() || frame.cols != windowWidth || frame.rows != windowHeight) {
        frame = cv::Mat(windowHeight, windowWidth, CV_8UC3, cv::Scalar(40, 40, 50));
    } else {
        frame = cv::Scalar(40, 40, 50);
    }
    
    switch (currentMode) {
        case AppMode::MODE_SELECT:
            renderModeSelect(frame);
            break;
        case AppMode::SOURCE_SELECT:
            renderSourceSelect(frame);
            break;
        case AppMode::MAZE_SELECT:
            renderMazeSelect(frame);
            break;
        case AppMode::CALIBRATION:
            renderCalibration(frame);
            break;
        case AppMode::PLAYING:
            // Game rendering handled elsewhere
            break;
    }
}

void GUIManager::handleMouse(int event, int x, int y) {
    // Update hover states
    for (auto& btn : buttons) {
        btn.isHovered = btn.contains(x, y) && btn.isEnabled;
    }
    
    // Handle clicks
    if (event == cv::EVENT_LBUTTONDOWN) {
        for (auto& btn : buttons) {
            if (btn.isHovered && btn.onClick) {
                btn.onClick();
                return;
            }
        }
    }
    
    // Handle scroll
    if (event == cv::EVENT_MOUSEWHEEL) {
        // Scroll logic if needed
    }
}

bool GUIManager::handleKey(int key) {
    if (key == 27 || key == 'q' || key == 'Q') {  // ESC or Q
        if (onQuit) onQuit();
        return true;
    }
    
    if (key == 'r' || key == 'R') {  // Reset
        if (currentMode == AppMode::PLAYING && onResetGame) {
            onResetGame();
            return true;
        }
    }
    
    if (key == 8 || key == 127) {  // Backspace
        // Go back to previous menu
        if (currentMode == AppMode::SOURCE_SELECT) {
            setMode(AppMode::MODE_SELECT);
            return true;
        } else if (currentMode == AppMode::MAZE_SELECT) {
            setMode(AppMode::SOURCE_SELECT);
            return true;
        }
    }
    
    return false;
}

void GUIManager::update() {
    // Any periodic updates
}

void GUIManager::setMode(AppMode mode) {
    currentMode = mode;
    buttons.clear();
    scrollOffset = 0;
    
    switch (mode) {
        case AppMode::MODE_SELECT:
            createModeSelectButtons();
            break;
        case AppMode::SOURCE_SELECT:
            createSourceSelectButtons();
            break;
        case AppMode::MAZE_SELECT:
            createMazeSelectButtons();
            break;
        case AppMode::CALIBRATION:
            createCalibrationButtons();
            break;
        default:
            break;
    }
}

void GUIManager::setImageFiles(const std::vector<std::string>& files) {
    imageFiles = files;
}

void GUIManager::setVideoFiles(const std::vector<std::string>& files) {
    videoFiles = files;
}

void GUIManager::setWebcams(const std::vector<int>& cams) {
    webcams = cams;
}

void GUIManager::setYamlFiles(const std::vector<std::string>& files) {
    yamlFiles = files;
}

void GUIManager::resetSelection() {
    selectedSource.clear();
    selectedYaml.clear();
    selectedWebcam = -1;
    selectionComplete = false;
    setMode(AppMode::MODE_SELECT);
}

void GUIManager::createModeSelectButtons() {
    buttons.clear();
    
    int btnWidth = 300;
    int btnHeight = 60;
    int startY = windowHeight / 2 - 100;
    int centerX = windowWidth / 2 - btnWidth / 2;
    int spacing = 80;
    
    // Image mode button
    Button imgBtn(centerX, startY, btnWidth, btnHeight, "IMAGE MODE");
    imgBtn.normalColor = cv::Scalar(60, 100, 60);
    imgBtn.hoverColor = cv::Scalar(80, 140, 80);
    imgBtn.onClick = [this]() {
        sourceType = "image";
        setMode(AppMode::SOURCE_SELECT);
    };
    buttons.push_back(imgBtn);
    
    // Video mode button
    Button vidBtn(centerX, startY + spacing, btnWidth, btnHeight, "VIDEO MODE");
    vidBtn.normalColor = cv::Scalar(60, 60, 100);
    vidBtn.hoverColor = cv::Scalar(80, 80, 140);
    vidBtn.onClick = [this]() {
        sourceType = "video";
        setMode(AppMode::SOURCE_SELECT);
    };
    buttons.push_back(vidBtn);
    
    // Webcam mode button
    Button camBtn(centerX, startY + spacing * 2, btnWidth, btnHeight, "WEBCAM MODE");
    camBtn.normalColor = cv::Scalar(100, 60, 60);
    camBtn.hoverColor = cv::Scalar(140, 80, 80);
    camBtn.onClick = [this]() {
        sourceType = "webcam";
        setMode(AppMode::SOURCE_SELECT);
    };
    buttons.push_back(camBtn);
    
    // Quit button
    Button quitBtn(centerX, startY + spacing * 3 + 20, btnWidth, btnHeight, "QUIT");
    quitBtn.normalColor = cv::Scalar(80, 40, 40);
    quitBtn.hoverColor = cv::Scalar(120, 60, 60);
    quitBtn.onClick = [this]() {
        if (onQuit) onQuit();
    };
    buttons.push_back(quitBtn);
}

void GUIManager::createSourceSelectButtons() {
    buttons.clear();
    
    int btnWidth = 350;
    int btnHeight = 45;
    int startY = 120;
    int centerX = windowWidth / 2 - btnWidth / 2;
    int spacing = 55;
    
    std::vector<std::string>* sourceList = nullptr;
    
    if (sourceType == "image") {
        sourceList = &imageFiles;
    } else if (sourceType == "video") {
        sourceList = &videoFiles;
    }
    
    if (sourceType == "webcam") {
        // Webcam buttons
        for (size_t i = 0; i < webcams.size() && i < 5; ++i) {
            std::ostringstream ss;
            ss << "Webcam " << webcams[i];
            Button btn(centerX, startY + (int)i * spacing, btnWidth, btnHeight, ss.str());
            btn.normalColor = cv::Scalar(70, 70, 90);
            btn.hoverColor = cv::Scalar(100, 100, 130);
            int camIdx = webcams[i];
            btn.onClick = [this, camIdx]() {
                selectedWebcam = camIdx;
                setMode(AppMode::MAZE_SELECT);
            };
            buttons.push_back(btn);
        }
        
        if (webcams.empty()) {
            // No webcams found - show helpful message
            Button btn(centerX, startY, btnWidth, btnHeight, "No webcams detected");
            btn.isEnabled = false;
            btn.normalColor = cv::Scalar(50, 50, 50);
            buttons.push_back(btn);
            
            // Add tip
            Button tipBtn(centerX, startY + spacing, btnWidth, btnHeight, "Check: ls /dev/video*");
            tipBtn.isEnabled = false;
            tipBtn.normalColor = cv::Scalar(40, 40, 40);
            buttons.push_back(tipBtn);
        }
    } else if (sourceList) {
        // File buttons
        for (size_t i = 0; i < sourceList->size() && i < 8; ++i) {
            Button btn(centerX, startY + (int)i * spacing, btnWidth, btnHeight, (*sourceList)[i]);
            btn.normalColor = cv::Scalar(70, 70, 90);
            btn.hoverColor = cv::Scalar(100, 100, 130);
            std::string filename = (*sourceList)[i];
            btn.onClick = [this, filename]() {
                selectedSource = filename;
                setMode(AppMode::MAZE_SELECT);
            };
            buttons.push_back(btn);
        }
        
        if (sourceList->empty()) {
            std::string msg = "No " + sourceType + " files found";
            Button btn(centerX, startY, btnWidth, btnHeight, msg);
            btn.isEnabled = false;
            btn.normalColor = cv::Scalar(50, 50, 50);
            buttons.push_back(btn);
            
            // Add tip for where to put files
            std::string tipMsg = "Add files to data/" + sourceType + "/";
            Button tipBtn(centerX, startY + spacing, btnWidth, btnHeight, tipMsg);
            tipBtn.isEnabled = false;
            tipBtn.normalColor = cv::Scalar(40, 40, 40);
            buttons.push_back(tipBtn);
            
            // Show supported formats
            std::string formatMsg = sourceType == "video" ? 
                "Formats: .mp4, .avi, .mov, .mkv" : 
                "Formats: .jpg, .png, .bmp";
            Button formatBtn(centerX, startY + spacing * 2, btnWidth, btnHeight, formatMsg);
            formatBtn.isEnabled = false;
            formatBtn.normalColor = cv::Scalar(40, 40, 40);
            buttons.push_back(formatBtn);
        }
    }
    
    // Back button
    Button backBtn(20, windowHeight - 60, 100, 40, "< Back");
    backBtn.normalColor = cv::Scalar(60, 60, 60);
    backBtn.hoverColor = cv::Scalar(90, 90, 90);
    backBtn.onClick = [this]() {
        setMode(AppMode::MODE_SELECT);
    };
    buttons.push_back(backBtn);
}

void GUIManager::createMazeSelectButtons() {
    buttons.clear();
    
    int btnWidth = 250;
    int btnHeight = 50;
    int startY = 150;
    int centerX = windowWidth / 2 - btnWidth / 2;
    int spacing = 65;
    
    auto mazeTypes = MazeGenerator::getAllMazeTypes();
    
    for (size_t i = 0; i < mazeTypes.size(); ++i) {
        std::string name = MazeGenerator::getMazeTypeName(mazeTypes[i]);
        Button btn(centerX, startY + (int)i * spacing, btnWidth, btnHeight, name);
        
        // Color based on difficulty
        switch (mazeTypes[i]) {
            case MazeType::SIMPLE:
                btn.normalColor = cv::Scalar(60, 120, 60);
                btn.hoverColor = cv::Scalar(80, 160, 80);
                break;
            case MazeType::MEDIUM:
                btn.normalColor = cv::Scalar(120, 120, 60);
                btn.hoverColor = cv::Scalar(160, 160, 80);
                break;
            case MazeType::HARD:
                btn.normalColor = cv::Scalar(120, 60, 60);
                btn.hoverColor = cv::Scalar(160, 80, 80);
                break;
            default:
                btn.normalColor = cv::Scalar(60, 60, 120);
                btn.hoverColor = cv::Scalar(80, 80, 160);
        }
        
        MazeType mt = mazeTypes[i];
        btn.onClick = [this, mt]() {
            selectedMaze = mt;
            selectionComplete = true;
            currentMode = AppMode::PLAYING;
            if (onStartGame) onStartGame();
        };
        buttons.push_back(btn);
    }
    
    // Back button
    Button backBtn(20, windowHeight - 60, 100, 40, "< Back");
    backBtn.normalColor = cv::Scalar(60, 60, 60);
    backBtn.hoverColor = cv::Scalar(90, 90, 90);
    backBtn.onClick = [this]() {
        setMode(AppMode::SOURCE_SELECT);
    };
    buttons.push_back(backBtn);
}

void GUIManager::createCalibrationButtons() {
    buttons.clear();
    // Calibration mode buttons if needed
}

void GUIManager::drawButton(cv::Mat& frame, const Button& btn) {
    cv::Scalar color = btn.isHovered ? btn.hoverColor : btn.normalColor;
    if (!btn.isEnabled) {
        color = cv::Scalar(50, 50, 50);
    }
    
    // Button background
    cv::rectangle(frame, btn.bounds, color, -1);
    
    // Button border
    cv::Scalar borderColor = btn.isHovered ? cv::Scalar(200, 200, 200) : cv::Scalar(100, 100, 100);
    cv::rectangle(frame, btn.bounds, borderColor, 2);
    
    // Button text
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(btn.label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    int textX = btn.bounds.x + (btn.bounds.width - textSize.width) / 2;
    int textY = btn.bounds.y + (btn.bounds.height + textSize.height) / 2;
    
    cv::Scalar textColor = btn.isEnabled ? btn.textColor : cv::Scalar(120, 120, 120);
    cv::putText(frame, btn.label, cv::Point(textX, textY),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2, cv::LINE_AA);
}

void GUIManager::drawTitle(cv::Mat& frame, const std::string& title) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(title, cv::FONT_HERSHEY_DUPLEX, 1.5, 3, &baseline);
    int textX = (windowWidth - textSize.width) / 2;
    
    cv::putText(frame, title, cv::Point(textX, 60),
                cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
}

void GUIManager::drawSubtitle(cv::Mat& frame, const std::string& subtitle, int yOffset) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(subtitle, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    int textX = (windowWidth - textSize.width) / 2;
    
    cv::putText(frame, subtitle, cv::Point(textX, yOffset),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(180, 180, 180), 2, cv::LINE_AA);
}

void GUIManager::renderModeSelect(cv::Mat& frame) {
    drawTitle(frame, "AR MAZE GAME");
    drawSubtitle(frame, "Select Input Mode", 100);
    
    for (const auto& btn : buttons) {
        drawButton(frame, btn);
    }
    
    // Footer with tips
    cv::putText(frame, "Use your webcam or select an image/video with a flat surface",
                cv::Point(windowWidth / 2 - 250, windowHeight - 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(150, 150, 150), 1, cv::LINE_AA);
    cv::putText(frame, "Tip: Run with video directly: ./ARMazeGame path/to/video.mp4",
                cv::Point(windowWidth / 2 - 220, windowHeight - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(120, 150, 120), 1, cv::LINE_AA);
}

void GUIManager::renderSourceSelect(cv::Mat& frame) {
    std::string title = "Select " + sourceType;
    title[7] = std::toupper(title[7]);  // Capitalize first letter of type
    drawTitle(frame, title);
    drawSubtitle(frame, "Choose your input source", 100);
    
    for (const auto& btn : buttons) {
        drawButton(frame, btn);
    }
}

void GUIManager::renderMazeSelect(cv::Mat& frame) {
    drawTitle(frame, "Select Maze");
    drawSubtitle(frame, "Choose difficulty level", 100);
    
    for (const auto& btn : buttons) {
        drawButton(frame, btn);
    }
    
    // Show selected source info
    std::string sourceInfo;
    if (sourceType == "webcam") {
        sourceInfo = "Using: Webcam " + std::to_string(selectedWebcam);
    } else {
        sourceInfo = "Using: " + selectedSource;
    }
    cv::putText(frame, sourceInfo, cv::Point(20, windowHeight - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 150), 1, cv::LINE_AA);
}

void GUIManager::renderCalibration(cv::Mat& frame) {
    drawTitle(frame, "Camera Calibration");
    drawSubtitle(frame, "Follow the instructions", 100);
    
    for (const auto& btn : buttons) {
        drawButton(frame, btn);
    }
}

void GUIManager::renderInGameHUD(cv::Mat& frame) {
    // In-game HUD rendering - handled by MazeRenderer
}