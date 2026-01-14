/**
 * @file RAII.hpp
 * @brief RAII wrappers for resource management (GLFW, OpenGL, OpenCV)
 * @author Olivier Deruelle
 * @date 2025
 */

#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>
#include <functional>
#include <iostream>
#include <algorithm>

/**
 * @brief RAII wrapper for GLFW window management
 */
class GLFWManager {
private:
    GLFWwindow* window;
    bool initialized;

public:
    GLFWManager(int width, int height, const char* title);
    ~GLFWManager();
    
    // Non-copyable
    GLFWManager(const GLFWManager&) = delete;
    GLFWManager& operator=(const GLFWManager&) = delete;
    
    GLFWwindow* getWindow() const { return window; }
    bool shouldClose() const;
    void swapBuffers();
    void pollEvents();
    void getFramebufferSize(int& width, int& height);
};

/**
 * @brief RAII wrapper for OpenGL shaders
 */
class GLShader {
private:
    GLuint shader_id;
    bool valid;

public:
    GLShader(GLenum type, const char* source);
    ~GLShader();
    
    // Non-copyable
    GLShader(const GLShader&) = delete;
    GLShader& operator=(const GLShader&) = delete;
    
    GLuint getId() const { return shader_id; }
    bool isValid() const { return valid; }
};

/**
 * @brief RAII wrapper for OpenGL programs
 */
class GLProgram {
private:
    GLuint program_id;
    bool valid;

public:
    GLProgram(const std::vector<GLuint>& shaders);
    ~GLProgram();
    
    // Non-copyable
    GLProgram(const GLProgram&) = delete;
    GLProgram& operator=(const GLProgram&) = delete;
    
    void use() const;
    GLint getUniformLocation(const char* name) const;
    bool isValid() const { return valid; }
};

/**
 * @brief RAII wrapper for OpenGL textures
 */
class GLTexture {
private:
    GLuint texture_id;
    int width, height, channels;
    bool valid;

public:
    GLTexture(int w, int h, int ch);
    ~GLTexture();
    
    // Non-copyable
    GLTexture(const GLTexture&) = delete;
    GLTexture& operator=(const GLTexture&) = delete;
    
    void bind(int unit = 0) const;
    void unbind() const;
    void updateFromMat(const cv::Mat& img);
    bool isValid() const { return valid; }
};

/**
 * @brief RAII wrapper for OpenGL mesh (VAO/VBO/EBO)
 */
class MeshRAII {
private:
    GLuint vao, vbo, ebo;
    GLsizei vertex_count;
    bool has_ebo;
    bool valid;
    
    void cleanup();

public:
    MeshRAII();
    ~MeshRAII();
    
    // Non-copyable
    MeshRAII(const MeshRAII&) = delete;
    MeshRAII& operator=(const MeshRAII&) = delete;
    
    // Moveable
    MeshRAII(MeshRAII&& other) noexcept;
    MeshRAII& operator=(MeshRAII&& other) noexcept;
    
    template<typename LayoutFunc>
    void initVertexOnly(const void* vertices, size_t vertex_size, GLsizei count, LayoutFunc layout) {
        cleanup();
        
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertex_size, vertices, GL_STATIC_DRAW);
        
        layout(); // Setup vertex attributes
        
        glBindVertexArray(0);
        
        vertex_count = count;
        has_ebo = false;
        valid = true;
    }
    
    template<typename LayoutFunc>
    void initWithIndices(const void* vertices, size_t vertex_size, 
                        const void* indices, size_t index_size, GLsizei count, 
                        LayoutFunc layout) {
        cleanup();
        
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);
        
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertex_size, vertices, GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_size, indices, GL_STATIC_DRAW);
        
        layout(); // Setup vertex attributes
        
        glBindVertexArray(0);
        
        vertex_count = count;
        has_ebo = true;
        valid = true;
    }
    
    void bind() const;
    void unbind() const;
    GLsizei count() const { return vertex_count; }
    bool hasIndices() const { return has_ebo; }
    bool isValid() const { return valid; }
};

/**
 * @brief RAII wrapper for axis mesh (simpler vertex-only mesh)
 */
class AxisMeshRAII {
private:
    GLuint vao, vbo;
    GLsizei vertex_count;
    bool valid;
    
    void cleanup();

public:
    AxisMeshRAII();
    ~AxisMeshRAII();
    
    // Non-copyable
    AxisMeshRAII(const AxisMeshRAII&) = delete;
    AxisMeshRAII& operator=(const AxisMeshRAII&) = delete;
    
    // Moveable
    AxisMeshRAII(AxisMeshRAII&& other) noexcept;
    AxisMeshRAII& operator=(AxisMeshRAII&& other) noexcept;
    
    void init(const void* vertices, size_t size, GLsizei count);
    void bind() const;
    void unbind() const;
    GLsizei count() const { return vertex_count; }
    bool isValid() const { return valid; }
};

/**
 * @brief RAII wrapper for OpenCV VideoCapture
 */
class VideoCaptureRAII {
private:
    cv::VideoCapture cap;
    bool initialized;

public:
    explicit VideoCaptureRAII(const std::string& source);
    ~VideoCaptureRAII();
    
    // Non-copyable
    VideoCaptureRAII(const VideoCaptureRAII&) = delete;
    VideoCaptureRAII& operator=(const VideoCaptureRAII&) = delete;
    
    bool isOpened() const;
    bool read(cv::Mat& frame);
    double get(int propId) const;
};

/**
 * @brief Resource cleanup manager for systematic resource management
 */
class ResourceManager {
private:
    std::vector<std::function<void()>> cleanup_functions;
    bool cleaned_up;

public:
    ResourceManager();
    ~ResourceManager();
    
    void registerCleanup(std::function<void()> cleanup_func);
    void cleanup();
    
    // Non-copyable
    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;
};
