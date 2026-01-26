#include "Maze.hpp"
#include <cmath>

MazeGenerator::MazeGenerator() {}

std::vector<WallSegment> MazeGenerator::generateMaze(MazeType type, 
                                                      float halfW, float halfH, 
                                                      float wallThickness) {
    switch (type) {
        case MazeType::LEVEL1:
            return generateLevel1Maze(halfW, halfH, wallThickness);
        case MazeType::LEVEL2:
            return generateLevel2Maze(halfW, halfH, wallThickness);
        default:
            return generateLevel1Maze(halfW, halfH, wallThickness);
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
        case MazeType::LEVEL1:
        case MazeType::LEVEL2:
            // Goal at center for both levels
            g.position = Vec2(0, 0);
            break;
        default:
            g.position = Vec2(0, 0);
    }
    
    return g;
}

std::string MazeGenerator::getMazeTypeName(MazeType type) {
    switch (type) {
        case MazeType::LEVEL1: return "Level 1";
        case MazeType::LEVEL2: return "Level 2";
        default: return "Unknown";
    }
}

std::vector<MazeType> MazeGenerator::getAllMazeTypes() {
    return {
        MazeType::LEVEL1,
        MazeType::LEVEL2
    };
}

// Level 1 : spiraling maze
std::vector<WallSegment> MazeGenerator::generateLevel1Maze(float halfW, float halfH, float thickness) {
    std::vector<WallSegment> walls;
    
    walls.emplace_back(-halfW * 1.0f, halfH * 0.7f, halfW * 0.6f, halfH * 0.7f, thickness);
    walls.emplace_back(halfW * 0.6f, halfH * 0.7f, halfW * 0.6f, -halfH * 0.7f, thickness);
    walls.emplace_back(halfW * 0.6f, -halfH * 0.7f, -halfW * 0.6f, -halfH * 0.7f, thickness);
    walls.emplace_back(-halfW * 0.6f, -halfH * 0.7f, -halfW * 0.6f, halfH * 0.4f, thickness);
    walls.emplace_back(-halfW * 0.6f, halfH * 0.4f, halfW * 0.3f, halfH * 0.4f, thickness);
    walls.emplace_back(halfW * 0.3f, halfH * 0.4f, halfW * 0.3f, -halfH * 0.4f, thickness);
    walls.emplace_back(halfW * 0.3f, -halfH * 0.4f, -halfW * 0.2f, -halfH * 0.4f, thickness);



    return walls;
}

// Level 2: more openings -> more difficult
std::vector<WallSegment> MazeGenerator::generateLevel2Maze(float halfW, float halfH, float thickness) {
    std::vector<WallSegment> walls;

    
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

// ============= GameState Implementation =============

GameState::GameState()
    : currentState(State::MENU)
    , currentMaze(MazeType::LEVEL1)
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