/**
 * @file RAII.cpp
 * @brief Implementation of RAII resource management classes
 */

#include "RAII.hpp"
#include <iostream>
#include <algorithm>

// ============= GLFWManager =============
GLFWManager::GLFWManager(int width, int height, const char* title) 
    : window(nullptr), initialized(false) {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    initialized = true;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // 4x MSAA
    
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // V-Sync
}

GLFWManager::~GLFWManager() {
    if (window) {
        glfwDestroyWindow(window);
    }
    if (initialized) {
        glfwTerminate();
    }
}

bool GLFWManager::shouldClose() const {
    return glfwWindowShouldClose(window);
}

void GLFWManager::swapBuffers() {
    glfwSwapBuffers(window);
}

void GLFWManager::pollEvents() {
    glfwPollEvents();
}

void GLFWManager::getFramebufferSize(int& width, int& height) {
    glfwGetFramebufferSize(window, &width, &height);
}

// ============= GLShader =============
GLShader::GLShader(GLenum type, const char* source) : shader_id(0), valid(false) {
    shader_id = glCreateShader(type);
    glShaderSource(shader_id, 1, &source, nullptr);
    glCompileShader(shader_id);
    
    GLint success;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
    if (success) {
        valid = true;
    } else {
        GLint logLength;
        glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<GLchar> log(std::max(1, logLength));
        glGetShaderInfoLog(shader_id, logLength, nullptr, log.data());
        std::cerr << "Shader compilation error: " << log.data() << std::endl;
        glDeleteShader(shader_id);
        shader_id = 0;
    }
}

GLShader::~GLShader() {
    if (shader_id != 0) {
        glDeleteShader(shader_id);
    }
}

// ============= GLProgram =============
GLProgram::GLProgram(const std::vector<GLuint>& shaders) : program_id(0), valid(false) {
    program_id = glCreateProgram();
    for (GLuint shader : shaders) {
        glAttachShader(program_id, shader);
    }
    glLinkProgram(program_id);
    
    GLint success;
    glGetProgramiv(program_id, GL_LINK_STATUS, &success);
    if (success) {
        valid = true;
    } else {
        GLint logLength;
        glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<GLchar> log(std::max(1, logLength));
        glGetProgramInfoLog(program_id, logLength, nullptr, log.data());
        std::cerr << "Program linking error: " << log.data() << std::endl;
        glDeleteProgram(program_id);
        program_id = 0;
    }
}

GLProgram::~GLProgram() {
    if (program_id != 0) {
        glDeleteProgram(program_id);
    }
}

void GLProgram::use() const {
    if (valid) glUseProgram(program_id);
}

GLint GLProgram::getUniformLocation(const char* name) const {
    return valid ? glGetUniformLocation(program_id, name) : -1;
}

// ============= GLTexture =============
GLTexture::GLTexture(int w, int h, int ch) 
    : texture_id(0), width(w), height(h), channels(ch), valid(false) {
    glGenTextures(1, &texture_id);
    if (texture_id != 0) {
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        
        GLint internalFormat = GL_RGB8;
        GLenum format = GL_RGB;
        if (channels == 4) { internalFormat = GL_RGBA8; format = GL_RGBA; }
        else if (channels == 1) { internalFormat = GL_R8; format = GL_RED; }
        
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, w, h, 0, format, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
        valid = true;
    }
}

GLTexture::~GLTexture() {
    if (texture_id != 0) {
        glDeleteTextures(1, &texture_id);
    }
}

void GLTexture::bind(int unit) const {
    if (valid) {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, texture_id);
    }
}

void GLTexture::unbind() const {
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLTexture::updateFromMat(const cv::Mat& img) {
    if (!valid || img.cols != width || img.rows != height) return;
    
    glBindTexture(GL_TEXTURE_2D, texture_id);
    GLenum format = (img.channels() == 4) ? GL_RGBA : ((img.channels() == 3) ? GL_RGB : GL_RED);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.cols, img.rows, format, GL_UNSIGNED_BYTE, img.data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// ============= MeshRAII =============
MeshRAII::MeshRAII() : vao(0), vbo(0), ebo(0), vertex_count(0), has_ebo(false), valid(false) {}

MeshRAII::~MeshRAII() {
    cleanup();
}

MeshRAII::MeshRAII(MeshRAII&& other) noexcept 
    : vao(other.vao), vbo(other.vbo), ebo(other.ebo), 
      vertex_count(other.vertex_count), has_ebo(other.has_ebo), valid(other.valid) {
    other.vao = other.vbo = other.ebo = 0;
    other.valid = false;
}

MeshRAII& MeshRAII::operator=(MeshRAII&& other) noexcept {
    if (this != &other) {
        cleanup();
        vao = other.vao;
        vbo = other.vbo;
        ebo = other.ebo;
        vertex_count = other.vertex_count;
        has_ebo = other.has_ebo;
        valid = other.valid;
        other.vao = other.vbo = other.ebo = 0;
        other.valid = false;
    }
    return *this;
}

void MeshRAII::bind() const {
    if (valid) glBindVertexArray(vao);
}

void MeshRAII::unbind() const {
    glBindVertexArray(0);
}

void MeshRAII::cleanup() {
    if (vao != 0) { glDeleteVertexArrays(1, &vao); vao = 0; }
    if (vbo != 0) { glDeleteBuffers(1, &vbo); vbo = 0; }
    if (ebo != 0) { glDeleteBuffers(1, &ebo); ebo = 0; }
    valid = false;
}

// ============= AxisMeshRAII =============
AxisMeshRAII::AxisMeshRAII() : vao(0), vbo(0), vertex_count(0), valid(false) {}

AxisMeshRAII::~AxisMeshRAII() {
    cleanup();
}

AxisMeshRAII::AxisMeshRAII(AxisMeshRAII&& other) noexcept 
    : vao(other.vao), vbo(other.vbo), vertex_count(other.vertex_count), valid(other.valid) {
    other.vao = other.vbo = 0;
    other.valid = false;
}

AxisMeshRAII& AxisMeshRAII::operator=(AxisMeshRAII&& other) noexcept {
    if (this != &other) {
        cleanup();
        vao = other.vao;
        vbo = other.vbo;
        vertex_count = other.vertex_count;
        valid = other.valid;
        other.vao = other.vbo = 0;
        other.valid = false;
    }
    return *this;
}

void AxisMeshRAII::init(const void* vertices, size_t size, GLsizei count) {
    cleanup();
    
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    
    glBindVertexArray(0);
    
    vertex_count = count;
    valid = true;
}

void AxisMeshRAII::bind() const {
    if (valid) glBindVertexArray(vao);
}

void AxisMeshRAII::unbind() const {
    glBindVertexArray(0);
}

void AxisMeshRAII::cleanup() {
    if (vao != 0) { glDeleteVertexArrays(1, &vao); vao = 0; }
    if (vbo != 0) { glDeleteBuffers(1, &vbo); vbo = 0; }
    valid = false;
}

// ============= VideoCaptureRAII =============
VideoCaptureRAII::VideoCaptureRAII(const std::string& source) : initialized(false) {
    if (source == "0") {
        cap.open(0);
    } else {
        cap.open(source);
    }
    initialized = cap.isOpened();
}

VideoCaptureRAII::~VideoCaptureRAII() {
    if (initialized) {
        cap.release();
    }
}

bool VideoCaptureRAII::isOpened() const {
    return initialized && cap.isOpened();
}

bool VideoCaptureRAII::read(cv::Mat& frame) {
    return cap.read(frame);
}

double VideoCaptureRAII::get(int propId) const {
    return cap.get(propId);
}

// ============= ResourceManager =============
ResourceManager::ResourceManager() : cleaned_up(false) {}

ResourceManager::~ResourceManager() {
    cleanup();
}

void ResourceManager::registerCleanup(std::function<void()> cleanup_func) {
    cleanup_functions.push_back(cleanup_func);
}

void ResourceManager::cleanup() {
    if (!cleaned_up) {
        for (auto it = cleanup_functions.rbegin(); it != cleanup_functions.rend(); ++it) {
            try {
                (*it)();
            } catch (const std::exception& e) {
                std::cerr << "Cleanup error: " << e.what() << std::endl;
            }
        }
        cleanup_functions.clear();
        cleaned_up = true;
    }
}
