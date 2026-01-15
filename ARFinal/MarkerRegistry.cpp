/**
 * @file MarkerRegistry.cpp
 * @brief Implementation of extensible marker and 3D object registry
 */

#include "MarkerRegistry.hpp"
#include "Detection.hpp"
#include "Rendering.hpp"
#include <iostream>
#include <fstream>

// ============= CubeObject =============
CubeObject::CubeObject(float cubeSize, const glm::vec3& cubeColor) 
    : size(cubeSize), color(cubeColor) {
    mesh = createCubeWireframeRAII(size / 2.0f, size / 2.0f);
}

void CubeObject::render(const glm::mat4& mvp, const GLProgram& program) {
    program.use();
    GLint loc_mvp = program.getUniformLocation("uMVP");
    GLint loc_color = program.getUniformLocation("uColor");
    
    glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform3f(loc_color, color.r, color.g, color.b);
    
    mesh.bind();
    glDrawElements(GL_LINES, mesh.count(), GL_UNSIGNED_INT, 0);
    mesh.unbind();
}

// ============= PyramidObject =============
PyramidObject::PyramidObject(float base, float h, const glm::vec3& pyramidColor)
    : baseSize(base), height(h), color(pyramidColor) {
    
    // Create pyramid mesh
    float half = baseSize / 2.0f;
    const float vertices[] = {
        // Base square (z=0)
        -half, -half, 0.0f,
         half, -half, 0.0f,
         half,  half, 0.0f,
        -half,  half, 0.0f,
        // Apex
        0.0f, 0.0f, -height
    };
    
    const GLuint indices[] = {
        // Base
        0, 1, 1, 2, 2, 3, 3, 0,
        // Edges to apex
        0, 4, 1, 4, 2, 4, 3, 4
    };
    
    mesh.initWithIndices(vertices, sizeof(vertices), indices, sizeof(indices),
                        sizeof(indices)/sizeof(indices[0]), []() {
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    });
}

void PyramidObject::render(const glm::mat4& mvp, const GLProgram& program) {
    program.use();
    GLint loc_mvp = program.getUniformLocation("uMVP");
    GLint loc_color = program.getUniformLocation("uColor");
    
    glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform3f(loc_color, color.r, color.g, color.b);
    
    mesh.bind();
    glDrawElements(GL_LINES, mesh.count(), GL_UNSIGNED_INT, 0);
    mesh.unbind();
}

// ============= SphereObject =============
SphereObject::SphereObject(float r, int segs, const glm::vec3& sphereColor)
    : radius(r), segments(segs), color(sphereColor) {
    
    // Generate sphere wireframe (latitude/longitude lines)
    std::vector<float> vertices;
    std::vector<GLuint> indices;
    
    // Generate vertices
    for (int lat = 0; lat <= segments; ++lat) {
        float theta = lat * M_PI / segments;
        float sinTheta = std::sin(theta);
        float cosTheta = std::cos(theta);
        
        for (int lon = 0; lon <= segments; ++lon) {
            float phi = lon * 2 * M_PI / segments;
            float sinPhi = std::sin(phi);
            float cosPhi = std::cos(phi);
            
            float x = cosPhi * sinTheta;
            float y = sinPhi * sinTheta;
            float z = cosTheta;
            
            vertices.push_back(x * radius);
            vertices.push_back(y * radius);
            vertices.push_back(-z * radius); // Negative Z for AR convention
        }
    }
    
    // Generate indices for wireframe
    for (int lat = 0; lat < segments; ++lat) {
        for (int lon = 0; lon < segments; ++lon) {
            int first = lat * (segments + 1) + lon;
            int second = first + segments + 1;
            
            // Latitude lines
            indices.push_back(first);
            indices.push_back(first + 1);
            
            // Longitude lines
            indices.push_back(first);
            indices.push_back(second);
        }
    }
    
    mesh.initWithIndices(vertices.data(), vertices.size() * sizeof(float),
                        indices.data(), indices.size() * sizeof(GLuint),
                        indices.size(), []() {
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    });
}

void SphereObject::render(const glm::mat4& mvp, const GLProgram& program) {
    program.use();
    GLint loc_mvp = program.getUniformLocation("uMVP");
    GLint loc_color = program.getUniformLocation("uColor");
    
    glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform3f(loc_color, color.r, color.g, color.b);
    
    mesh.bind();
    glDrawElements(GL_LINES, mesh.count(), GL_UNSIGNED_INT, 0);
    mesh.unbind();
}

// ============= A4MarkerDetector =============
bool A4MarkerDetector::detect(const cv::Mat& frame, 
                              std::vector<cv::Point2f>& corners,
                              double& confidence) {
    return detectA4CornersAdaptive(frame, corners, confidence);
}

// ============= ArUcoMarkerDetector =============
ArUcoMarkerDetector::ArUcoMarkerDetector(int id, float size)
    : markerId(id), markerSize(size) {
    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
}

bool ArUcoMarkerDetector::detect(const cv::Mat& frame, 
                                 std::vector<cv::Point2f>& corners,
                                 double& confidence) {
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    
    // Ancienne API OpenCV 4.6 (pas besoin de créer un ArucoDetector)
    cv::aruco::detectMarkers(frame, dictionary, markerCorners, ids);
    
    // Find our specific marker
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == markerId) {
            corners = markerCorners[i];
            confidence = 0.9; // ArUco detection is generally very reliable
            return true;
        }
    }
    
    confidence = 0.0;
    return false;
}

// ============= MarkerRegistry =============
void MarkerRegistry::registerPair(const std::string& name,
                                  std::shared_ptr<MarkerDetector> marker,
                                  std::shared_ptr<Object3D> object) {
    registry[name] = MarkerObjectPair(marker, object, name);
    std::cout << "✓ Registered: " << name << " (" << marker->getType() 
              << " → " << object->getName() << ")\n";
}

void MarkerRegistry::unregisterPair(const std::string& name) {
    auto it = registry.find(name);
    if (it != registry.end()) {
        std::cout << "✗ Unregistered: " << name << "\n";
        registry.erase(it);
    }
}

void MarkerRegistry::setEnabled(const std::string& name, bool enabled) {
    auto it = registry.find(name);
    if (it != registry.end()) {
        it->second.enabled = enabled;
        std::cout << (enabled ? "✓ Enabled: " : "✗ Disabled: ") << name << "\n";
    }
}

bool MarkerRegistry::loadFromConfig(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Cannot open marker registry config: " << filename << "\n";
        return false;
    }
    
    cv::FileNode markers = fs["markers"];
    for (auto it = markers.begin(); it != markers.end(); ++it) {
        std::string name = (*it)["name"];
        std::string markerType = (*it)["marker_type"];
        std::string objectType = (*it)["object_type"];
        bool enabled = (int)(*it)["enabled"] != 0;
        
        // Parse marker parameters
        std::map<std::string, float> markerParams;
        cv::FileNode mParams = (*it)["marker_params"];
        for (auto p = mParams.begin(); p != mParams.end(); ++p) {
            std::string key = (*p).name();
            markerParams[key] = (float)(*p);
        }
        
        // Parse object parameters
        std::map<std::string, float> objectParams;
        cv::FileNode oParams = (*it)["object_params"];
        for (auto p = oParams.begin(); p != oParams.end(); ++p) {
            std::string key = (*p).name();
            objectParams[key] = (float)(*p);
        }
        
        // Parse color
        glm::vec3 color(1.0f, 1.0f, 1.0f);
        if (!(*it)["color"].empty()) {
            cv::FileNode colorNode = (*it)["color"];
            color.r = (float)colorNode[0];
            color.g = (float)colorNode[1];
            color.b = (float)colorNode[2];
        }
        
        // Create marker and object
        auto marker = MarkerFactory::create(markerType, markerParams);
        auto object = ObjectFactory::create(objectType, objectParams, color);
        
        if (marker && object) {
            registerPair(name, marker, object);
            setEnabled(name, enabled);
        }
    }
    
    std::cout << "Loaded " << registry.size() << " marker-object pairs from " << filename << "\n";
    return true;
}

bool MarkerRegistry::saveToConfig(const std::string& filename) const {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Cannot save marker registry config: " << filename << "\n";
        return false;
    }
    
    fs << "markers" << "[";
    for (const auto& pair : registry) {
        fs << "{";
        fs << "name" << pair.second.name;
        fs << "enabled" << (pair.second.enabled ? 1 : 0);
        fs << "marker_type" << pair.second.marker->getType();
        fs << "object_type" << pair.second.object->getName();
        fs << "}";
    }
    fs << "]";
    
    std::cout << "Saved " << registry.size() << " marker-object pairs to " << filename << "\n";
    return true;
}

void MarkerRegistry::printRegistry() const {
    std::cout << "\n=== MARKER REGISTRY ===\n";
    std::cout << "Total pairs: " << registry.size() << "\n\n";
    
    for (const auto& pair : registry) {
        std::cout << (pair.second.enabled ? "✓" : "✗") << " " << pair.first << "\n";
        std::cout << "  Marker: " << pair.second.marker->getType() << "\n";
        std::cout << "  Object: " << pair.second.object->getName() 
                  << " - " << pair.second.object->getDescription() << "\n";
        auto dims = pair.second.marker->getDimensions();
        std::cout << "  Dimensions: " << dims.width << "x" << dims.height << " mm\n\n";
    }
    std::cout << "======================\n\n";
}

// ============= ObjectFactory =============
std::shared_ptr<Object3D> ObjectFactory::create(const std::string& type,
                                               const std::map<std::string, float>& params,
                                               const glm::vec3& color) {
    if (type == "Cube") {
        float size = params.count("size") ? params.at("size") : 50.0f;
        return std::make_shared<CubeObject>(size, color);
    }
    else if (type == "Pyramid") {
        float base = params.count("base") ? params.at("base") : 60.0f;
        float height = params.count("height") ? params.at("height") : 80.0f;
        return std::make_shared<PyramidObject>(base, height, color);
    }
    else if (type == "Sphere") {
        float radius = params.count("radius") ? params.at("radius") : 30.0f;
        int segments = params.count("segments") ? (int)params.at("segments") : 16;
        return std::make_shared<SphereObject>(radius, segments, color);
    }
    
    std::cerr << "Unknown object type: " << type << "\n";
    return nullptr;
}

// ============= MarkerFactory =============
std::shared_ptr<MarkerDetector> MarkerFactory::create(const std::string& type,
                                                      const std::map<std::string, float>& params) {
    if (type == "A4 Paper") {
        return std::make_shared<A4MarkerDetector>();
    }
    else if (type == "ArUco") {
        int id = params.count("id") ? (int)params.at("id") : 0;
        float size = params.count("size") ? params.at("size") : 100.0f;
        return std::make_shared<ArUcoMarkerDetector>(id, size);
    }
    
    std::cerr << "Unknown marker type: " << type << "\n";
    return nullptr;
}
