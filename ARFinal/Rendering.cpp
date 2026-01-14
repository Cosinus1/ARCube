/**
 * @file Rendering.cpp
 * @brief Implementation of OpenGL rendering utilities
 */

#include "Rendering.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>

// ============= Shader Sources =============
const char* BG_VS = R"(#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
out vec2 vUV;
void main(){ vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }
)";

const char* BG_FS = R"(#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
void main(){ FragColor = texture(uTex, vUV); }
)";

const char* LINE_VS = R"(#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main(){ gl_Position = uMVP * vec4(aPos, 1.0); }
)";

const char* LINE_GS = R"(#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform float uThicknessPx;
uniform vec2  uViewport;
void main(){
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec2 ndc0 = p0.xy / p0.w;
    vec2 ndc1 = p1.xy / p1.w;
    vec2 dir = ndc1 - ndc0;
    float len = length(dir);
    vec2 n = (len > 1e-6) ? normalize(vec2(-dir.y, dir.x)) : vec2(0.0, 1.0);
    vec2 px2ndc = 2.0 / uViewport;
    vec2 off = n * uThicknessPx * px2ndc;
    float z0 = p0.z / p0.w;
    float z1 = p1.z / p1.w;
    vec4 v0 = vec4(ndc0 - off, z0, 1.0);
    vec4 v1 = vec4(ndc0 + off, z0, 1.0);
    vec4 v2 = vec4(ndc1 - off, z1, 1.0);
    vec4 v3 = vec4(ndc1 + off, z1, 1.0);
    gl_Position = v0; EmitVertex();
    gl_Position = v1; EmitVertex();
    gl_Position = v2; EmitVertex();
    gl_Position = v3; EmitVertex();
    EndPrimitive();
}
)";

const char* LINE_FS = R"(#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main(){ FragColor = vec4(uColor, 1.0); }
)";

const char* MARKER_VS = R"(#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
uniform float uPointSize;
void main(){
    gl_Position = uMVP * vec4(aPos,1.0);
    gl_PointSize = uPointSize;
}
)";

const char* MARKER_FS = R"(#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main(){
    vec2 c = gl_PointCoord*2.0-1.0;
    float d = dot(c,c);
    if(d>1.0) discard;
    float alpha = smoothstep(1.0,0.8,d);
    FragColor = vec4(uColor, alpha);
}
)";

const char* PLANE_VS = R"(#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main(){ gl_Position = uMVP * vec4(aPos,1.0); }
)";

const char* PLANE_FS = R"(#version 330 core
out vec4 FragColor;
uniform vec4 uPlaneColor;
void main(){ FragColor = uPlaneColor; }
)";

// Lit plane with subtle edge shadowing
const char* PLANE_LIGHT_VS = R"(#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
uniform mat4 uMV;
out vec3 vViewPos;
out vec2 vLocal;
void main(){
    vec4 viewPos = uMV * vec4(aPos,1.0);
    vViewPos = viewPos.xyz;
    vLocal = aPos.xy;
    gl_Position = uMVP * vec4(aPos,1.0);
}
)";

const char* PLANE_LIGHT_FS = R"(#version 330 core
in vec3 vViewPos;
in vec2 vLocal;
out vec4 FragColor;
uniform vec4 uColor;
uniform vec2 uHalfExtents;

void main(){
    // Per-fragment normal from derivatives in view space
    vec3 n = normalize(cross(dFdx(vViewPos), dFdy(vViewPos)));
    vec3 L = normalize(-vViewPos);
    float diff = max(dot(n, L), 0.0);
    float ambient = 0.45;
    float shade = ambient + 0.55 * diff;

    // Soft shadow toward the walls based on local coordinates
    float hx = uHalfExtents.x;
    float hy = uHalfExtents.y;
    float minEdge = min(hx - abs(vLocal.x), hy - abs(vLocal.y));
    float falloff = min(hx, hy) * 0.25;
    float occlusion = smoothstep(0.0, falloff, minEdge);
    float edgeShadow = mix(0.55, 1.0, occlusion);

    float finalShade = shade * edgeShadow;
    FragColor = vec4(uColor.rgb * finalShade, uColor.a);
}
)";

// Simple lighting for walls (camera light)
const char* WALL_LIGHT_VS = R"(#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
uniform mat4 uMV;
out vec3 vViewPos;
void main(){
    vec4 viewPos = uMV * vec4(aPos,1.0);
    vViewPos = viewPos.xyz;
    gl_Position = uMVP * vec4(aPos,1.0);
}
)";

const char* WALL_LIGHT_FS = R"(#version 330 core
in vec3 vViewPos;
out vec4 FragColor;
uniform vec4 uColor;

void main(){
    // Derive normal in view space
    vec3 n = normalize(cross(dFdx(vViewPos), dFdy(vViewPos)));
    // Light from camera (origin in view space)
    vec3 L = normalize(-vViewPos);
    float diff = max(dot(n, L), 0.0);
    float ambient = 0.32;
    float rim = pow(1.0 - diff, 2.0) * 0.18; // subtle rim to accent edges
    float shade = ambient + 0.55 * diff + rim;
    FragColor = vec4(uColor.rgb * shade, uColor.a);
}
)";

// Smoothly lit sphere (uses vertex normals)
const char* SPHERE_VS = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uMV;
uniform mat3 uNormalMat;
out vec3 vNormal;
out vec3 vViewPos;
void main(){
    vec4 viewPos = uMV * vec4(aPos,1.0);
    vViewPos = viewPos.xyz;
    vNormal = normalize(uNormalMat * aNormal);
    gl_Position = uMVP * vec4(aPos,1.0);
}
)";

const char* SPHERE_FS = R"(#version 330 core
in vec3 vNormal;
in vec3 vViewPos;
out vec4 FragColor;
uniform vec4 uColor;
uniform float uShininess;

void main(){
    vec3 N = normalize(vNormal);
    vec3 L = normalize(-vViewPos); // camera at origin in view space
    vec3 V = normalize(-vViewPos);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), uShininess);
    float ambient = 0.25;
    float shade = ambient + 0.65 * diff + 0.2 * spec;
    FragColor = vec4(uColor.rgb * shade, uColor.a);
}
)";

// ============= Shader Compilation =============
GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint logLen = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &logLen);
        std::vector<GLchar> log(std::max(1, logLen));
        glGetShaderInfoLog(s, logLen, nullptr, log.data());
        std::cerr << "Shader compile error: " << log.data() << std::endl;
        glDeleteShader(s);
        throw std::runtime_error("Shader compile failed");
    }
    return s;
}

GLuint linkProgram(const std::vector<GLuint>& shaders) {
    GLuint p = glCreateProgram();
    for (auto s : shaders) glAttachShader(p, s);
    glLinkProgram(p);
    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint logLen = 0;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &logLen);
        std::vector<GLchar> log(std::max(1, logLen));
        glGetProgramInfoLog(p, logLen, nullptr, log.data());
        std::cerr << "Program link error:\n" << log.data() << std::endl;
        glDeleteProgram(p);
        throw std::runtime_error("Program link failed");
    }
    return p;
}

// ============= Matrix Utilities =============
glm::mat4 projectionFromCV(const cv::Mat& K, float w, float h, float n, float f) {
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    glm::mat4 P(0.0f);
    P[0][0] = static_cast<float>(2.0 * fx / w);
    P[1][1] = static_cast<float>(2.0 * fy / h);
    P[2][0] = static_cast<float>(2.0 * cx / w - 1.0);
    P[2][1] = static_cast<float>(1.0 - 2.0 * cy / h);
    P[2][2] = static_cast<float>(-(f + n) / (f - n));
    P[3][2] = static_cast<float>(-2.0 * f * n / (f - n));
    P[2][3] = -1.0f;

    return P;
}

glm::mat4 getModelViewMatrix(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);

    glm::mat4 M(1.0f);
    
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            M[c][r] = static_cast<float>(Rcv.at<double>(r, c));
        }
    }
    
    M[3][0] = static_cast<float>(tvec.at<double>(0));
    M[3][1] = static_cast<float>(tvec.at<double>(1));
    M[3][2] = static_cast<float>(tvec.at<double>(2));
    M[3][3] = 1.0f;

    glm::mat4 cvToGl(1.0f);
    cvToGl[1][1] = -1.0f;
    cvToGl[2][2] = -1.0f;

    return cvToGl * M;
}

// ============= Mesh Creation =============
MeshRAII createBackgroundQuadRAII() {
    float data[] = {
        -1.f, -1.f,     0.f, 0.f,
         1.f, -1.f,     1.f, 0.f,
         1.f,  1.f,     1.f, 1.f,
        -1.f, -1.f,     0.f, 0.f,
         1.f,  1.f,     1.f, 1.f,
        -1.f,  1.f,     0.f, 1.f
    };
    
    MeshRAII mesh;
    mesh.initVertexOnly(data, sizeof(data), 6, []() {
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
    });
    
    return mesh;
}

MeshRAII createCubeWireframeRAII(float half, float h) {
    float top = -h;
    const float vertices[] = {
        -half, -half, 0.f,  +half, -half, 0.f,  +half, +half, 0.f,  -half, +half, 0.f,
        -half, -half, top,  +half, -half, top,  +half, +half, top,  -half, +half, top
    };
    const GLuint indices[] = {
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7
    };
    
    MeshRAII mesh;
    mesh.initWithIndices(vertices, sizeof(vertices), indices, sizeof(indices), 
                        static_cast<GLsizei>(sizeof(indices)/sizeof(indices[0])), []() {
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    });
    
    return mesh;
}

std::vector<MeshRAII> createWallsRAII(float halfW, float halfH, float h, float thickness) {
    std::vector<MeshRAII> walls;
    float top = -h;
    float thick = thickness;
    
    // Rectangular sheet corners (detected A4)
    glm::vec3 corners[] = {
        {-halfW, -halfH, 0.f},  // BL
        {+halfW, -halfH, 0.f},  // BR
        {+halfW, +halfH, 0.f},  // TR
        {-halfW, +halfH, 0.f}   // TL
    };
    
    // Build 4 walls around the rectangle
    for (int w = 0; w < 4; ++w) {
        int c1 = w;
        int c2 = (w + 1) % 4;
        glm::vec3 p1 = corners[c1];
        glm::vec3 p2 = corners[c2];
        
        glm::vec3 edge = p2 - p1;
        glm::vec3 normal = glm::normalize(glm::vec3(-edge.y, edge.x, 0.f));
        
        glm::vec3 p1_out = p1 + normal * thick;
        glm::vec3 p2_out = p2 + normal * thick;
        
        glm::vec3 p1_top     = {p1.x,     p1.y,     top};
        glm::vec3 p2_top     = {p2.x,     p2.y,     top};
        glm::vec3 p1_out_top = {p1_out.x, p1_out.y, top};
        glm::vec3 p2_out_top = {p2_out.x, p2_out.y, top};
        
        float wallVertices[] = {
            // bottom ring
            p1.x, p1.y, p1.z,
            p2.x, p2.y, p2.z,
            p2_out.x, p2_out.y, p2_out.z,
            p1_out.x, p1_out.y, p1_out.z,
            // top ring
            p1_top.x, p1_top.y, p1_top.z,
            p2_top.x, p2_top.y, p2_top.z,
            p2_out_top.x, p2_out_top.y, p2_out_top.z,
            p1_out_top.x, p1_out_top.y, p1_out_top.z
        };
        
        GLuint wallIndices[] = {
            0, 1, 5,  0, 5, 4, // inner
            2, 3, 7,  2, 7, 6, // outer
            0, 4, 7,  0, 7, 3, // left
            1, 6, 5,  1, 2, 6, // right
            4, 5, 6,  4, 6, 7, // top cap
            0, 3, 2,  0, 2, 1  // bottom cap
        };
        
        MeshRAII wallMesh;
        wallMesh.initWithIndices(wallVertices, sizeof(wallVertices),
                                wallIndices, sizeof(wallIndices),
                                static_cast<GLsizei>(sizeof(wallIndices)/sizeof(wallIndices[0])),
                                []() {
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
        });
        walls.push_back(std::move(wallMesh));
    }
    return walls;
}

MeshRAII createSphereRAII(float radius, int slices, int stacks) {
    std::vector<float> data;
    std::vector<GLuint> indices;
    data.reserve((size_t)(slices + 1) * (stacks + 1) * 6);
    indices.reserve((size_t)slices * stacks * 6);

    for (int i = 0; i <= stacks; ++i) {
        float v = (float)i / (float)stacks;
        float phi = v * glm::pi<float>();
        float y = cosf(phi);
        float r = sinf(phi);
        for (int j = 0; j <= slices; ++j) {
            float u = (float)j / (float)slices;
            float theta = u * glm::two_pi<float>();
            float x = r * cosf(theta);
            float z = r * sinf(theta);
            glm::vec3 n = glm::normalize(glm::vec3(x, y, z));
            glm::vec3 p = n * radius;
            data.insert(data.end(), {p.x, p.y, p.z, n.x, n.y, n.z});
        }
    }

    int stride = slices + 1;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int i0 = i * stride + j;
            int i1 = i0 + 1;
            int i2 = i0 + stride;
            int i3 = i2 + 1;
            indices.insert(indices.end(), { (GLuint)i0, (GLuint)i2, (GLuint)i1 });
            indices.insert(indices.end(), { (GLuint)i1, (GLuint)i2, (GLuint)i3 });
        }
    }

    MeshRAII mesh;
    mesh.initWithIndices(data.data(), data.size() * sizeof(float),
                         indices.data(), indices.size() * sizeof(GLuint),
                         static_cast<GLsizei>(indices.size()), [](){
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    });
    return mesh;
}

std::vector<MeshRAII> createMazeWallsRAII(float halfW, float halfH, float h, float thickness) {
    std::vector<MeshRAII> mazeWalls;
    float top = -h;
    
    // Helper: build a straight wall segment from (x1,y1) to (x2,y2)
    auto buildWall = [&](float x1, float y1, float x2, float y2) {
        glm::vec3 p1(x1, y1, 0.f);
        glm::vec3 p2(x2, y2, 0.f);
        glm::vec3 edge = p2 - p1;
        glm::vec3 normal = glm::normalize(glm::vec3(-edge.y, edge.x, 0.f));
        glm::vec3 p1_out = p1 + normal * thickness;
        glm::vec3 p2_out = p2 + normal * thickness;
        glm::vec3 p1_top(p1.x, p1.y, top);
        glm::vec3 p2_top(p2.x, p2.y, top);
        glm::vec3 p1_out_top(p1_out.x, p1_out.y, top);
        glm::vec3 p2_out_top(p2_out.x, p2_out.y, top);
        
        float wallVertices[] = {
            p1.x, p1.y, p1.z,
            p2.x, p2.y, p2.z,
            p2_out.x, p2_out.y, p2_out.z,
            p1_out.x, p1_out.y, p1_out.z,
            p1_top.x, p1_top.y, p1_top.z,
            p2_top.x, p2_top.y, p2_top.z,
            p2_out_top.x, p2_out_top.y, p2_out_top.z,
            p1_out_top.x, p1_out_top.y, p1_out_top.z
        };
        GLuint wallIndices[] = {
            0, 1, 5,  0, 5, 4,
            2, 3, 7,  2, 7, 6,
            0, 4, 7,  0, 7, 3,
            1, 6, 5,  1, 2, 6,
            4, 5, 6,  4, 6, 7,
            0, 3, 2,  0, 2, 1
        };
        MeshRAII wallMesh;
        wallMesh.initWithIndices(wallVertices, sizeof(wallVertices),
                                wallIndices, sizeof(wallIndices),
                                static_cast<GLsizei>(sizeof(wallIndices)/sizeof(wallIndices[0])),
                                []() {
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
        });
        mazeWalls.push_back(std::move(wallMesh));
    };

    // Simpler maze layout - cross pattern with openings
    // Central vertical wall with gap in middle
    buildWall(0.f, halfH * 0.8f, 0.f, halfH * 0.2f);
    buildWall(0.f, -halfH * 0.2f, 0.f, -halfH * 0.8f);
    
    // Central horizontal wall with gap in middle
    buildWall(-halfW * 0.8f, 0.f, -halfW * 0.2f, 0.f);
    buildWall(halfW * 0.2f, 0.f, halfW * 0.8f, 0.f);
    
    // Few extra obstacles
    buildWall(-halfW * 0.5f, halfH * 0.5f, halfW * 0.0f, halfH * 0.5f);
    buildWall(halfW * 0.0f, -halfH * 0.5f, halfW * 0.5f, -halfH * 0.5f);
    
    return mazeWalls;
}

AxisMeshRAII createAxesRAII(float len) {
    float vertices[] = {
        0.f, 0.f, 0.f,   len, 0.f, 0.f,   // X
        0.f, 0.f, 0.f,   0.f, len, 0.f,   // Y
        0.f, 0.f, 0.f,   0.f, 0.f, -len   // Z
    };
    
    AxisMeshRAII mesh;
    mesh.init(vertices, sizeof(vertices), 6);
    return mesh;
}
