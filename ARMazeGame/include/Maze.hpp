#pragma once
#include "Physics.hpp"
#include <vector>
#include <string>

/**
 * @brief Maze layout types
 */
enum class MazeType {
    SIMPLE,         // Easy maze for beginners
    MEDIUM,         // Medium difficulty
    HARD,           // Challenging maze
    SPIRAL,         // Spiral pattern
    ZIGZAG          // Zigzag pattern
};

/**
 * @brief Maze configuration and generation
 */
class MazeGenerator {
public:
    MazeGenerator();
    
    /**
     * @brief Generate maze walls for the physics simulator
     * @param type Maze difficulty/pattern type
     * @param halfW Half width of play area
     * @param halfH Half height of play area
     * @param wallThickness Thickness of walls
     * @return Vector of wall segments
     */
    std::vector<WallSegment> generateMaze(MazeType type, 
                                          float halfW, float halfH, 
                                          float wallThickness);
    
    /**
     * @brief Get starting position for ball
     */
    Vec2 getStartPosition(MazeType type, float halfW, float halfH);
    
    /**
     * @brief Get goal position
     */
    Goal getGoalPosition(MazeType type, float halfW, float halfH);
    
    /**
     * @brief Get maze type name as string
     */
    static std::string getMazeTypeName(MazeType type);
    
    /**
     * @brief Get all available maze types
     */
    static std::vector<MazeType> getAllMazeTypes();

private:
    std::vector<WallSegment> generateSimpleMaze(float halfW, float halfH, float thickness);
    std::vector<WallSegment> generateMediumMaze(float halfW, float halfH, float thickness);
    std::vector<WallSegment> generateHardMaze(float halfW, float halfH, float thickness);
    std::vector<WallSegment> generateSpiralMaze(float halfW, float halfH, float thickness);
    std::vector<WallSegment> generateZigzagMaze(float halfW, float halfH, float thickness);
};

/**
 * @brief Game state management
 */
class GameState {
public:
    enum class State {
        MENU,           // Main menu
        PLAYING,        // Active gameplay
        WIN,            // Player won
        PAUSED          // Game paused
    };
    
    GameState();
    
    void startGame(MazeType maze);
    void pauseGame();
    void resumeGame();
    void resetGame();
    void win();
    
    State getState() const { return currentState; }
    MazeType getCurrentMaze() const { return currentMaze; }
    float getElapsedTime() const;
    int getAttempts() const { return attempts; }
    
    void update(float deltaTime);

private:
    State currentState;
    MazeType currentMaze;
    float elapsedTime;
    int attempts;
    bool timerRunning;
};
