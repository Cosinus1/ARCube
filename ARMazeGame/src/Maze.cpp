#include "Maze.hpp"
#include <cmath>

MazeGenerator::MazeGenerator() {}

std::vector<WallSegment> MazeGenerator::generateMaze(MazeType type, 
                                                      float halfW, float halfH, 
                                                      float wallThickness) {
    switch (type) {
        case MazeType::SIMPLE:
            return generateSimpleMaze(halfW, halfH, wallThickness);
        case MazeType::MEDIUM:
            return generateMediumMaze(halfW, halfH, wallThickness);
        case MazeType::HARD:
            return generateHardMaze(halfW, halfH, wallThickness);
        case MazeType::SPIRAL:
            return generateSpiralMaze(halfW, halfH, wallThickness);
        case MazeType::ZIGZAG:
            return generateZigzagMaze(halfW, halfH, wallThickness);
        default:
            return generateSimpleMaze(halfW, halfH, wallThickness);
    }
}

Vec2 MazeGenerator::getStartPosition(MazeType type, float halfW, float halfH) {
    // Start in top-left corner for all mazes
    return Vec2(-halfW * 0.75f, halfH * 0.75f);
}

Goal MazeGenerator::getGoalPosition(MazeType type, float halfW, float halfH) {
    Goal g;
    g.radius = std::min(halfW, halfH) * 0.12f;
    
    switch (type) {
        case MazeType::SIMPLE:
            // Goal at center
            g.position = Vec2(0, 0);
            break;
        case MazeType::MEDIUM:
            // Goal at bottom-right
            g.position = Vec2(halfW * 0.6f, -halfH * 0.6f);
            break;
        case MazeType::HARD:
            // Goal at center
            g.position = Vec2(0, 0);
            break;
        case MazeType::SPIRAL:
            // Goal at center of spiral
            g.position = Vec2(0, 0);
            break;
        case MazeType::ZIGZAG:
            // Goal at bottom-right
            g.position = Vec2(halfW * 0.7f, -halfH * 0.7f);
            break;
        default:
            g.position = Vec2(0, 0);
    }
    
    return g;
}

std::string MazeGenerator::getMazeTypeName(MazeType type) {
    switch (type) {
        case MazeType::SIMPLE: return "Simple";
        case MazeType::MEDIUM: return "Medium";
        case MazeType::HARD: return "Hard";
        case MazeType::SPIRAL: return "Spiral";
        case MazeType::ZIGZAG: return "Zigzag";
        default: return "Unknown";
    }
}

std::vector<MazeType> MazeGenerator::getAllMazeTypes() {
    return {
        MazeType::SIMPLE,
        MazeType::MEDIUM,
        MazeType::HARD,
        MazeType::SPIRAL,
        MazeType::ZIGZAG
    };
}

std::vector<WallSegment> MazeGenerator::generateSimpleMaze(float halfW, float halfH, float thickness) {
    std::vector<WallSegment> walls;
    
    // Simple cross pattern with openings
    // Vertical wall with gap in middle
    walls.emplace_back(0.f, halfH * 0.8f, 0.f, halfH * 0.2f, thickness);
    walls.emplace_back(0.f, -halfH * 0.2f, 0.f, -halfH * 0.8f, thickness);
    
    // Horizontal wall with gap in middle  
    walls.emplace_back(-halfW * 0.8f, 0.f, -halfW * 0.2f, 0.f, thickness);
    walls.emplace_back(halfW * 0.2f, 0.f, halfW * 0.8f, 0.f, thickness);
    
    return walls;
}

std::vector<WallSegment> MazeGenerator::generateMediumMaze(float halfW, float halfH, float thickness) {
    std::vector<WallSegment> walls;
    
    // Create a path from top-left to bottom-right with obstacles
    
    // Top-left area walls
    walls.emplace_back(-halfW * 0.5f, halfH * 0.5f, 0.f, halfH * 0.5f, thickness);
    walls.emplace_back(0.f, halfH * 0.5f, 0.f, halfH * 0.1f, thickness);
    
    // Middle barriers
    walls.emplace_back(-halfW * 0.8f, 0.f, -halfW * 0.2f, 0.f, thickness);
    walls.emplace_back(halfW * 0.2f, halfH * 0.2f, halfW * 0.2f, -halfH * 0.3f, thickness);
    
    // Bottom area walls
    walls.emplace_back(-halfW * 0.3f, -halfH * 0.4f, halfW * 0.3f, -halfH * 0.4f, thickness);
    walls.emplace_back(halfW * 0.5f, -halfH * 0.2f, halfW * 0.5f, -halfH * 0.8f, thickness);
    
    // Additional obstacles
    walls.emplace_back(-halfW * 0.6f, halfH * 0.2f, -halfW * 0.6f, -halfH * 0.2f, thickness);
    
    return walls;
}

std::vector<WallSegment> MazeGenerator::generateHardMaze(float halfW, float halfH, float thickness) {
    std::vector<WallSegment> walls;
    
    // Complex maze with multiple dead ends and narrow passages
    
    // Outer ring with openings
    walls.emplace_back(-halfW * 0.7f, halfH * 0.7f, halfW * 0.4f, halfH * 0.7f, thickness);
    walls.emplace_back(halfW * 0.7f, halfH * 0.7f, halfW * 0.7f, -halfH * 0.4f, thickness);
    walls.emplace_back(halfW * 0.7f, -halfH * 0.7f, -halfW * 0.4f, -halfH * 0.7f, thickness);
    walls.emplace_back(-halfW * 0.7f, -halfH * 0.7f, -halfW * 0.7f, halfH * 0.4f, thickness);
    
    // Inner barriers
    walls.emplace_back(-halfW * 0.4f, halfH * 0.4f, halfW * 0.1f, halfH * 0.4f, thickness);
    walls.emplace_back(halfW * 0.4f, halfH * 0.4f, halfW * 0.4f, halfH * 0.1f, thickness);
    walls.emplace_back(halfW * 0.4f, -halfH * 0.1f, halfW * 0.4f, -halfH * 0.4f, thickness);
    walls.emplace_back(-halfW * 0.1f, -halfH * 0.4f, halfW * 0.4f, -halfH * 0.4f, thickness);
    walls.emplace_back(-halfW * 0.4f, -halfH * 0.4f, -halfW * 0.4f, -halfH * 0.1f, thickness);
    walls.emplace_back(-halfW * 0.4f, halfH * 0.1f, -halfW * 0.4f, halfH * 0.4f, thickness);
    
    // Central obstacles
    walls.emplace_back(-halfW * 0.15f, halfH * 0.15f, halfW * 0.15f, halfH * 0.15f, thickness);
    walls.emplace_back(-halfW * 0.15f, -halfH * 0.15f, halfW * 0.15f, -halfH * 0.15f, thickness);
    
    return walls;
}

std::vector<WallSegment> MazeGenerator::generateSpiralMaze(float halfW, float halfH, float thickness) {
    std::vector<WallSegment> walls;
    
    // Spiral pattern leading to center
    float scale = 0.9f;
    
    // Outer spiral
    walls.emplace_back(-halfW * scale, halfH * scale, halfW * scale * 0.7f, halfH * scale, thickness);
    walls.emplace_back(halfW * scale, halfH * scale, halfW * scale, -halfH * scale * 0.7f, thickness);
    walls.emplace_back(halfW * scale, -halfH * scale, -halfW * scale * 0.7f, -halfH * scale, thickness);
    walls.emplace_back(-halfW * scale, -halfH * scale, -halfW * scale, halfH * scale * 0.5f, thickness);
    
    // Inner spiral
    scale = 0.5f;
    walls.emplace_back(-halfW * scale, halfH * scale, halfW * scale * 0.5f, halfH * scale, thickness);
    walls.emplace_back(halfW * scale, halfH * scale, halfW * scale, -halfH * scale * 0.3f, thickness);
    walls.emplace_back(halfW * scale, -halfH * scale, -halfW * scale * 0.3f, -halfH * scale, thickness);
    
    return walls;
}

std::vector<WallSegment> MazeGenerator::generateZigzagMaze(float halfW, float halfH, float thickness) {
    std::vector<WallSegment> walls;
    
    // Zigzag pattern from top to bottom
    float rowHeight = halfH * 0.4f;
    
    // Row 1 - from right
    walls.emplace_back(halfW * 0.8f, halfH * 0.6f, -halfW * 0.3f, halfH * 0.6f, thickness);
    
    // Row 2 - from left
    walls.emplace_back(-halfW * 0.8f, halfH * 0.2f, halfW * 0.3f, halfH * 0.2f, thickness);
    
    // Row 3 - from right
    walls.emplace_back(halfW * 0.8f, -halfH * 0.2f, -halfW * 0.3f, -halfH * 0.2f, thickness);
    
    // Row 4 - from left
    walls.emplace_back(-halfW * 0.8f, -halfH * 0.6f, halfW * 0.3f, -halfH * 0.6f, thickness);
    
    // Vertical connectors
    walls.emplace_back(-halfW * 0.5f, halfH * 0.6f, -halfW * 0.5f, halfH * 0.2f, thickness);
    walls.emplace_back(halfW * 0.5f, halfH * 0.2f, halfW * 0.5f, -halfH * 0.2f, thickness);
    walls.emplace_back(-halfW * 0.5f, -halfH * 0.2f, -halfW * 0.5f, -halfH * 0.6f, thickness);
    
    return walls;
}

// ============= GameState Implementation =============

GameState::GameState()
    : currentState(State::MENU)
    , currentMaze(MazeType::SIMPLE)
    , elapsedTime(0.0f)
    , attempts(0)
    , timerRunning(false)
{
}

void GameState::startGame(MazeType maze) {
    currentMaze = maze;
    currentState = State::PLAYING;
    elapsedTime = 0.0f;
    attempts++;
    timerRunning = true;
}

void GameState::pauseGame() {
    if (currentState == State::PLAYING) {
        currentState = State::PAUSED;
        timerRunning = false;
    }
}

void GameState::resumeGame() {
    if (currentState == State::PAUSED) {
        currentState = State::PLAYING;
        timerRunning = true;
    }
}

void GameState::resetGame() {
    elapsedTime = 0.0f;
    timerRunning = true;
    currentState = State::PLAYING;
}

void GameState::win() {
    currentState = State::WIN;
    timerRunning = false;
}

float GameState::getElapsedTime() const {
    return elapsedTime;
}

void GameState::update(float deltaTime) {
    if (timerRunning) {
        elapsedTime += deltaTime;
    }
}
