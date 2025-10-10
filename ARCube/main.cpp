// AR - 2025-2026
// S. Mavromatis
// Sequelette 3D
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <algorithm> // std::max

// -------------------- AR : Données/calibration --------------------
cv::Mat cameraMatrix, distCoeffs; // Paramètres intrinsèques
cv::Mat rvec, tvec; // Rotation, Translation

// -------------------- GL : shaders  ------------------------
static const char* BG_VS = R"(#version 330 core
layout (location = 0) in vec2 aPos;  // NDC [-1,1]
layout (location = 1) in vec2 aUV;

out vec2 vUV;

void main(){
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

static const char* BG_FS = R"(#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uTex;

void main(){
    FragColor = texture(uTex, vUV);
}
)";

static const char* LINE_VS = R"(#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 uMVP;

void main(){
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

// Geometry shader pour épaissir les lignes en pixels (écran)
static const char* LINE_GS = R"(#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform float uThicknessPx;   // épaisseur souhaitée (pixels)
uniform vec2  uViewport;      // taille framebuffer (pixels)

void main(){
    // Entrées clip-space du VS
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;

    // Vers NDC
    vec2 ndc0 = p0.xy / p0.w;
    vec2 ndc1 = p1.xy / p1.w;

    // Direction + normale écran
    vec2 dir = ndc1 - ndc0;
    float len = length(dir);
    vec2 n = (len > 1e-6) ? normalize(vec2(-dir.y, dir.x)) : vec2(0.0, 1.0);

    // Conversion pixels -> NDC
    vec2 px2ndc = 2.0 / uViewport; // 1px = 2/W ou 2/H
    vec2 off = n * uThicknessPx * px2ndc;

    // Conserver la profondeur (NDC z)
    float z0 = p0.z / p0.w;
    float z1 = p1.z / p1.w;

    // Quad épais en NDC
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

static const char* LINE_FS = R"(#version 330 core
out vec4 FragColor;
uniform vec3 uColor;

void main(){
    FragColor = vec4(uColor, 1.0);
}
)";

// -------------------- GL : utils shaders -------------------------------
static GLuint compileShader(GLenum type, const char* src) {
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

        std::cerr << "Shader compile error ("
                  << (type == GL_VERTEX_SHADER ? "vertex" : (type == GL_GEOMETRY_SHADER ? "geometry" : "fragment"))
                  << "):\n" << log.data() << std::endl;

        glDeleteShader(s);
        throw std::runtime_error("Shader compile failed");
    }
    return s;
}

static GLuint linkProgram(const std::vector<GLuint>& shaders) {
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

// -------------------- AR : Matrices ---------------------------
glm::mat4 projectionFromCV(const cv::Mat& K, float w, float h, float n, float f) {
    // K = [ fx  0  cx;  0  fy  cy;  0   0   1 ]
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);

    glm::mat4 P(0.0f);
    P[0][0] =  2.0f * static_cast<float>(fx) / w;
    P[1][1] =  2.0f * static_cast<float>(fy) / h;
    P[2][0] =  1.0f - 2.0f * static_cast<float>(cx) / w;
    P[2][1] =  2.0f * static_cast<float>(cy) / h - 1.0f;
    P[2][2] = -(f + n) / (f - n);
    P[2][3] = -1.0f;
    P[3][2] = -2.0f * f * n / (f - n);
    return P;
}

glm::mat4 getModelViewMatrix() {
    // OpenCV: X_cam = R * X_obj + t
    // OpenGL view: X_cam_gl = cvToGl * (R * X + t)
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat viewMatrix = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(viewMatrix(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(viewMatrix(cv::Rect(3, 0, 1, 3)));

    // OpenCV (x→, y↓, z→) -> OpenGL (x→, y↑, z←)
    cv::Mat cvToGl = (cv::Mat_<double>(4, 4) <<
         1,  0,  0, 0,
         0, -1,  0, 0,
         0,  0, -1, 0,
         0,  0,  0, 1);
    viewMatrix = cvToGl * viewMatrix;

    // Convertir en float + transposer pour GLM (column-major)
    cv::Mat view32f; viewMatrix.convertTo(view32f, CV_32F);
    cv::Mat view32fT = view32f.t();
    return glm::make_mat4(view32fT.ptr<float>());
}

// -------------------- OpenCV : calibration/pose ---------------
void loadCalibration(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) throw std::runtime_error("Impossible d'ouvrir " + filename);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
}

void estimatePose() {
    // Notre objet est un rectangle A4 (mm) posé dans le plan z=0
    std::vector<cv::Point3f> objectPoints = {
        {0, 0, 0}, {210, 0, 0}, {210, 297, 0}, {0, 297, 0}
    };

    // Ici les sommets de la feuille en coordonnées image
    std::vector<cv::Point2f> imagePoints = {
        {283, 88}, {156, 168}, {348, 320}, {472, 193}
    };
    
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
}

// -------------------- Géométries (VAO/VBO/EBO) ---------------
struct Mesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei count = 0; // number of indices or vertices
};

// Quad plein-écran (NDC) pour le fond texturé
Mesh createBackgroundQuad() {
    // 2 triangles (6 sommets), pos(NDC) + uv
    float data[] = {
        //   x     y      u    v
        -1.f, -1.f,     0.f, 0.f,
         1.f, -1.f,     1.f, 0.f,
         1.f,  1.f,     1.f, 1.f,

        -1.f, -1.f,     0.f, 0.f,
         1.f,  1.f,     1.f, 1.f,
        -1.f,  1.f,     0.f, 1.f
    };
    Mesh m; m.count = 6;
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); // aPos
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); // aUV
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
    glBindVertexArray(0);
    return m;
}

// Cube wireframe (taille en unités de ton monde, ex. 50)
Mesh createCubeWireframe(float size) {
    const float s = size;
    const float V[] = {
        -s,-s,-s,  +s,-s,-s,  +s,+s,-s,  -s,+s,-s, // 0..3 z-
        -s,-s,+s,  +s,-s,+s,  +s,+s,+s,  -s,+s,+s  // 4..7 z+
    };

    const GLuint E[] = {
        0,1, 1,2, 2,3, 3,0,    // base z-
        4,5, 5,6, 6,7, 7,4,    // base z+
        0,4, 1,5, 2,6, 3,7     // verticales
    };

    Mesh m; m.count = static_cast<GLsizei>(sizeof(E)/sizeof(E[0]));
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ebo);

    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(V), V, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(E), E, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0); // aPos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);

    glBindVertexArray(0);
    return m;
}

// Axes : trois segments de longueur L à partir de l'origine (0->L)
struct Axes {
    Mesh x, y, z;
};

Axes createAxes(float L) {
    Axes A{};

    auto makeLine = [](float x1, float y1, float z1,
                       float x2, float y2, float z2) {
        Mesh m;
        m.count = 2; // 2 sommets pour GL_LINES

        const float V[6] = { x1, y1, z1,  x2, y2, z2 };

        glGenVertexArrays(1, &m.vao);
        glGenBuffers(1, &m.vbo);

        glBindVertexArray(m.vao);
        glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(V), V, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0); // aPos
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              3 * sizeof(float), (void*)0);

        glBindVertexArray(0);
        return m;
    };

    A.x = makeLine(0.f, 0.f, 0.f,  L,  0.f, 0.f);
    A.y = makeLine(0.f, 0.f, 0.f,  0.f,  L,  0.f);
    A.z = makeLine(0.f, 0.f, 0.f,  0.f, 0.f,  L );
    return A;
}

// -------------------- Texture OpenCV -> OpenGL ----------------
GLuint createTextureFromMat(const cv::Mat& imgRGBorRGBA) {
    if (imgRGBorRGBA.empty()) throw std::runtime_error("Image OpenCV vide.");

    GLuint tex=0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    GLint internalFormat = GL_RGB8;
    GLenum format = GL_RGB;
    if (imgRGBorRGBA.channels() == 4) { internalFormat = GL_RGBA8; format = GL_RGBA; }
    else if (imgRGBorRGBA.channels() == 3) { internalFormat = GL_RGB8;  format = GL_RGB; }
    else if (imgRGBorRGBA.channels() == 1) { internalFormat = GL_R8;    format = GL_RED; }

    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat,
                 imgRGBorRGBA.cols, imgRGBorRGBA.rows, 0,
                 format, GL_UNSIGNED_BYTE, imgRGBorRGBA.data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

// -------------------- Programme principal ---------------------
int main() {
    try {
        std::cout << "Current path: " << std::filesystem::current_path() << std::endl;
        // Calcul du calibrage + PnP
        loadCalibration("data/camera.yaml");
        estimatePose();
        
        // L'image
        cv::Mat image = cv::imread("data/image_1.jpg", cv::IMREAD_UNCHANGED);
        if (image.empty()) { std::cerr << "Image not found!\n"; return -1; }
        if (image.channels() == 3) cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        else if (image.channels() == 4) cv::cvtColor(image, image, cv::COLOR_BGRA2RGBA);
        cv::flip(image, image, 0);

        // Init GLFW/GL
        if (!glfwInit()) return -1;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif
        const int W=640, H=480;
        GLFWwindow* window = glfwCreateWindow(W, H, "ARCube", nullptr, nullptr);
        if (!window) { glfwTerminate(); return -1; }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        glewExperimental = GL_TRUE; // core profile
        if (glewInit() != GLEW_OK) {
            std::cerr << "GLEW init failed\n";
            return -1;
        }
        glGetError(); // purge l'erreur GL potentielle de GLEW

        // --- Shaders ---
        // Shaders pour le fond (VS+FS), affichage de l'image sur un plan texturé
        GLuint bgVS = compileShader(GL_VERTEX_SHADER,   BG_VS);
        GLuint bgFS = compileShader(GL_FRAGMENT_SHADER, BG_FS);
        GLuint bgProgram = linkProgram({ bgVS, bgFS });
        glDeleteShader(bgVS); glDeleteShader(bgFS);

        // Shaders pour le tracé des lignes (cube + axes)
        GLuint lineVS = compileShader(GL_VERTEX_SHADER,   LINE_VS);
        GLuint lineGS = compileShader(GL_GEOMETRY_SHADER, LINE_GS);
        GLuint lineFS = compileShader(GL_FRAGMENT_SHADER, LINE_FS);
        GLuint lineProgram = linkProgram({ lineVS, lineGS, lineFS });
        glDeleteShader(lineVS); glDeleteShader(lineGS); glDeleteShader(lineFS);

        // Nos géométries : plan, cube, axes
        Mesh bg   = createBackgroundQuad();
        Mesh cube = createCubeWireframe(50.0f);
        Axes axes = createAxes(210.0f); // axes (X=R, Y=G, Z=B), 210 mm

        // Texture du fond, l'image
        GLuint bgTex = createTextureFromMat(image);

        // États GL
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.05f, 0.05f, 0.06f, 1.0f);

        // Matrices AR
        glm::mat4 projection = projectionFromCV(cameraMatrix, float(W), float(H), 0.1f, 1000.0f);
        glm::mat4 view       = getModelViewMatrix();
        glm::mat4 model      = glm::mat4(1.0f);

        // Uniform locations
        GLint bg_uTex         = glGetUniformLocation(bgProgram,   "uTex");
        GLint line_uMVP       = glGetUniformLocation(lineProgram, "uMVP");
        GLint line_uColor     = glGetUniformLocation(lineProgram, "uColor");
        GLint line_uThickness = glGetUniformLocation(lineProgram, "uThicknessPx");
        GLint line_uViewport  = glGetUniformLocation(lineProgram, "uViewport");

        const float THICKNESS_PX = 3.0f; // épaisseur des lignes pour le GS

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            int fbw, fbh; // taille framebuffer réelle (HiDPI inclus)
            glfwGetFramebufferSize(window, &fbw, &fbh);
            glViewport(0, 0, fbw, fbh);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Fond texturé (depth OFF), notre image
            glDisable(GL_DEPTH_TEST);
            glUseProgram(bgProgram);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, bgTex);
            glUniform1i(bg_uTex, 0);

            glBindVertexArray(bg.vao);
            glDrawArrays(GL_TRIANGLES, 0, bg.count);
            glBindVertexArray(0);
            glBindTexture(GL_TEXTURE_2D, 0);

            // Cube + Axes (depth ON) — lignes épaisses via GS
            glEnable(GL_DEPTH_TEST);
            glm::mat4 MVP = projection * view * model;

            glUseProgram(lineProgram);
            glUniformMatrix4fv(line_uMVP, 1, GL_FALSE, glm::value_ptr(MVP));
            glUniform2f(line_uViewport, (float)fbw, (float)fbh);
            glUniform1f(line_uThickness, THICKNESS_PX);

            // Cube, contours en noir
            glUniform3f(line_uColor, 0.f, 0.f, 0.f);
            glBindVertexArray(cube.vao);
            glDrawElements(GL_LINES, cube.count, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);

            // Axes en couleurs
            glBindVertexArray(axes.x.vao);
            glUniform3f(line_uColor, 1.f, 0.f, 0.f);
            glDrawArrays(GL_LINES, 0, axes.x.count);

            glBindVertexArray(axes.y.vao);
            glUniform3f(line_uColor, 0.f, 1.f, 0.f);
            glDrawArrays(GL_LINES, 0, axes.y.count);

            glBindVertexArray(axes.z.vao);
            glUniform3f(line_uColor, 0.f, 0.f, 1.f);
            glDrawArrays(GL_LINES, 0, axes.z.count);

            glBindVertexArray(0);

            glfwSwapBuffers(window);
        }

        // C'est fini !
        glDeleteProgram(bgProgram);
        glDeleteProgram(lineProgram);
        glDeleteTextures(1, &bgTex);

        glDeleteVertexArrays(1, &bg.vao);
        glDeleteBuffers(1, &bg.vbo);

        glDeleteVertexArrays(1, &cube.vao);
        glDeleteBuffers(1, &cube.vbo);
        glDeleteBuffers(1, &cube.ebo);

        glDeleteVertexArrays(1, &axes.x.vao);
        glDeleteBuffers(1, &axes.x.vbo);
        glDeleteVertexArrays(1, &axes.y.vao);
        glDeleteBuffers(1, &axes.y.vbo);
        glDeleteVertexArrays(1, &axes.z.vao);
        glDeleteBuffers(1, &axes.z.vbo);

        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return -1;
    }
}
