#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional>
#include "Maze.hpp"

/**
 * @brief Button structure for GUI
 */
struct Button {
    cv::Rect bounds;
    std::string label;
    cv::Scalar normalColor;
    cv::Scalar hoverColor;
    cv::Scalar textColor;
    bool isHovered;
    bool isEnabled;
    std::function<void()> onClick;
    
    Button() : isHovered(false), isEnabled(true),
               normalColor(80, 80, 80), hoverColor(120, 120, 120), 
               textColor(255, 255, 255) {}
    
    Button(int x, int y, int w, int h, const std::string& text)
        : bounds(x, y, w, h), label(text), isHovered(false), isEnabled(true),
          normalColor(80, 80, 80), hoverColor(120, 120, 120), 
          textColor(255, 255, 255) {}
    
    bool contains(int x, int y) const {
        return bounds.contains(cv::Point(x, y));
    }
};

/**
 * @brief Application modes
 */
enum class AppMode {
    MODE_SELECT,    // Select between image/video/webcam
    SOURCE_SELECT,  // Select specific source
    MAZE_SELECT,    // Select maze type
    CALIBRATION,    // Camera calibration
    PLAYING         // Active game
};

/**
 * @brief GUI Manager for the AR Maze Game
 */
class GUIManager {
public:
    GUIManager(int windowWidth, int windowHeight);
    
    /**
     * @brief Render the current GUI state
     * @param frame Output frame
     */
    void render(cv::Mat& frame);
    
    /**
     * @brief Handle mouse events
     * @param event Mouse event type
     * @param x Mouse X coordinate
     * @param y Mouse Y coordinate
     */
    void handleMouse(int event, int x, int y);
    
    /**
     * @brief Handle keyboard events
     * @param key Key code
     * @return true if event was handled
     */
    bool handleKey(int key);
    
    /**
     * @brief Update GUI state
     */
    void update();
    
    // Mode management
    AppMode getCurrentMode() const { return currentMode; }
    void setMode(AppMode mode);
    
    // Source management
    void setImageFiles(const std::vector<std::string>& files);
    void setVideoFiles(const std::vector<std::string>& files);
    void setWebcams(const std::vector<int>& cams);
    void setYamlFiles(const std::vector<std::string>& files);
    
    // Selection results
    std::string getSelectedSource() const { return selectedSource; }
    std::string getSelectedYaml() const { return selectedYaml; }
    int getSelectedWebcam() const { return selectedWebcam; }
    MazeType getSelectedMaze() const { return selectedMaze; }
    bool isSelectionComplete() const { return selectionComplete; }
    bool isImageMode() const { return sourceType == "image"; }
    bool isVideoMode() const { return sourceType == "video"; }
    bool isWebcamMode() const { return sourceType == "webcam"; }
    
    // Callbacks
    void setOnStartGame(std::function<void()> callback) { onStartGame = callback; }
    void setOnResetGame(std::function<void()> callback) { onResetGame = callback; }
    void setOnQuit(std::function<void()> callback) { onQuit = callback; }
    
    // Reset selection
    void resetSelection();

private:
    void renderModeSelect(cv::Mat& frame);
    void renderSourceSelect(cv::Mat& frame);
    void renderMazeSelect(cv::Mat& frame);
    void renderCalibration(cv::Mat& frame);
    void renderInGameHUD(cv::Mat& frame);
    
    void createModeSelectButtons();
    void createSourceSelectButtons();
    void createMazeSelectButtons();
    void createCalibrationButtons();
    
    void drawButton(cv::Mat& frame, const Button& btn);
    void drawTitle(cv::Mat& frame, const std::string& title);
    void drawSubtitle(cv::Mat& frame, const std::string& subtitle, int yOffset = 80);
    
    // Window dimensions
    int windowWidth, windowHeight;
    
    // Current state
    AppMode currentMode;
    std::string sourceType;
    std::string selectedSource;
    std::string selectedYaml;
    int selectedWebcam;
    MazeType selectedMaze;
    bool selectionComplete;
    
    // Available sources
    std::vector<std::string> imageFiles;
    std::vector<std::string> videoFiles;
    std::vector<int> webcams;
    std::vector<std::string> yamlFiles;
    
    // Buttons
    std::vector<Button> buttons;
    int scrollOffset;
    
    // Callbacks
    std::function<void()> onStartGame;
    std::function<void()> onResetGame;
    std::function<void()> onQuit;
};

/**
 * @brief Mouse callback wrapper
 */
void mouseCallback(int event, int x, int y, int flags, void* userdata);
