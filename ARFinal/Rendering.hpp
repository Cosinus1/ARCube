/**
 * @file Rendering.hpp
 * @brief OpenGL rendering utilities for AR application
 * @author Olivier Deruelle
 * @date 2025
 */

#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include "RAII.hpp"

// ============= Shader Sources =============
extern const char* BG_VS;
extern const char* BG_FS;
extern const char* LINE_VS;
extern const char* LINE_GS;
extern const char* LINE_FS;
extern const char* MARKER_VS;
extern const char* MARKER_FS;
extern const char* PLANE_VS;
extern const char* PLANE_FS;
extern const char* PLANE_LIGHT_VS;
extern const char* PLANE_LIGHT_FS;
extern const char* WALL_LIGHT_VS;
extern const char* WALL_LIGHT_FS;
extern const char* SPHERE_VS;
extern const char* SPHERE_FS;

// ============= Shader Compilation =============
/**
 * @brief Compiles a single OpenGL shader
 */
GLuint compileShader(GLenum type, const char* src);

/**
 * @brief Links multiple compiled shaders into a program
 */
GLuint linkProgram(const std::vector<GLuint>& shaders);

// ============= Matrix Utilities =============
/**
 * @brief Build OpenGL projection matrix from OpenCV intrinsics
 */
glm::mat4 projectionFromCV(const cv::Mat& K, float w, float h, float n, float f);

/**
 * @brief Builds OpenGL ModelView matrix from OpenCV pose
 */
glm::mat4 getModelViewMatrix(const cv::Mat& rvec, const cv::Mat& tvec);

// ============= Mesh Creation =============
/**
 * @brief Creates background quad mesh with RAII
 */
MeshRAII createBackgroundQuadRAII();

/**
 * @brief Creates cube wireframe mesh with RAII
 */
MeshRAII createCubeWireframeRAII(float half, float h);

/**
 * @brief Creates walls around the detected A4 plane perimeter with RAII
 * @param halfW half width of the sheet (A4_w/2)
 * @param halfH half height of the sheet (A4_h/2)
 * @param h wall height (positive value, wall is extruded to -h in Z)
 * @param thickness thickness of each wall outward from sheet edge
 */
std::vector<MeshRAII> createWallsRAII(float halfW, float halfH, float h, float thickness);

/**
 * @brief Creates a UV sphere mesh with positions and normals
 * @param radius sphere radius
 * @param slices longitudinal subdivisions
 * @param stacks latitudinal subdivisions
 */
MeshRAII createSphereRAII(float radius, int slices = 32, int stacks = 24);

/**
 * @brief Creates internal maze walls on the plane
 * @param halfW half width of the sheet
 * @param halfH half height of the sheet
 * @param h wall height (positive, extruded to -h)
 * @param thickness wall thickness
 */
std::vector<MeshRAII> createMazeWallsRAII(float halfW, float halfH, float h, float thickness);

/**
 * @brief Creates coordinate axes mesh with RAII
 */
AxisMeshRAII createAxesRAII(float len);
