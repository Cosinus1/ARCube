/**
 * @file MarkerRegistry.hpp
 * @brief Extensible marker and 3D object registry system
 * @author Olivier Deruelle
 * @date 2025
 * 
 * @details This module provides an extensible architecture for adding new markers
 * and 3D objects without modifying core code. Demonstrates excellent adaptability.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <functional>
#include "RAII.hpp"

/**
 * @brief Abstract base class for 3D objects
 * @details Allows easy extension with new object types
 */
class Object3D {
public:
    virtual ~Object3D() = default;
    
    /**
     * @brief Render the 3D object
     * @param mvp Model-View-Projection matrix
     * @param program OpenGL program to use
     */
    virtual void render(const glm::mat4& mvp, const GLProgram& program) = 0;
    
    /**
     * @brief Get object name
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get object description
     */
    virtual std::string getDescription() const = 0;
};

/**
 * @brief Cube wireframe object
 */
class CubeObject : public Object3D {
private:
    MeshRAII mesh;
    float size;
    glm::vec3 color;
    
public:
    CubeObject(float cubeSize, const glm::vec3& cubeColor);
    void render(const glm::mat4& mvp, const GLProgram& program) override;
    std::string getName() const override { return "Cube"; }
    std::string getDescription() const override { return "Wireframe cube"; }
};

/**
 * @brief Pyramid object (example of extensibility)
 */
class PyramidObject : public Object3D {
private:
    MeshRAII mesh;
    float baseSize;
    float height;
    glm::vec3 color;
    
public:
    PyramidObject(float base, float h, const glm::vec3& pyramidColor);
    void render(const glm::mat4& mvp, const GLProgram& program) override;
    std::string getName() const override { return "Pyramid"; }
    std::string getDescription() const override { return "3D pyramid"; }
};

/**
 * @brief Sphere object (example of extensibility)
 */
class SphereObject : public Object3D {
private:
    MeshRAII mesh;
    float radius;
    int segments;
    glm::vec3 color;
    
public:
    SphereObject(float r, int segs, const glm::vec3& sphereColor);
    void render(const glm::mat4& mvp, const GLProgram& program) override;
    std::string getName() const override { return "Sphere"; }
    std::string getDescription() const override { return "3D sphere"; }
};

/**
 * @brief Abstract base class for marker detectors
 * @details Allows adding new marker types (ArUco, QR, AprilTag, etc.)
 */
class MarkerDetector {
public:
    virtual ~MarkerDetector() = default;
    
    /**
     * @brief Detect marker in image
     * @param frame Input image
     * @param corners Output detected corners
     * @param confidence Output detection confidence
     * @return true if marker detected
     */
    virtual bool detect(const cv::Mat& frame, 
                       std::vector<cv::Point2f>& corners,
                       double& confidence) = 0;
    
    /**
     * @brief Get marker type name
     */
    virtual std::string getType() const = 0;
    
    /**
     * @brief Get marker dimensions (width, height in mm)
     */
    virtual cv::Size2f getDimensions() const = 0;
};

/**
 * @brief A4 paper marker detector (existing)
 */
class A4MarkerDetector : public MarkerDetector {
public:
    bool detect(const cv::Mat& frame, 
                std::vector<cv::Point2f>& corners,
                double& confidence) override;
    
    std::string getType() const override { return "A4 Paper"; }
    cv::Size2f getDimensions() const override { return cv::Size2f(210.0f, 297.0f); }
};

/**
 * @brief ArUco marker detector (extensibility example)
 */
class ArUcoMarkerDetector : public MarkerDetector {
private:
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    int markerId;
    float markerSize;
    
public:
    ArUcoMarkerDetector(int id, float size);
    
    bool detect(const cv::Mat& frame, 
                std::vector<cv::Point2f>& corners,
                double& confidence) override;
    
    std::string getType() const override { return "ArUco"; }
    cv::Size2f getDimensions() const override { return cv::Size2f(markerSize, markerSize); }
};

/**
 * @brief Marker-Object association
 */
struct MarkerObjectPair {
    std::shared_ptr<MarkerDetector> marker;
    std::shared_ptr<Object3D> object;
    std::string name;
    bool enabled;
    
    // Default constructor for std::map
    MarkerObjectPair() : enabled(false) {}
    
    MarkerObjectPair(std::shared_ptr<MarkerDetector> m,
                     std::shared_ptr<Object3D> o,
                     const std::string& n)
        : marker(m), object(o), name(n), enabled(true) {}
};

/**
 * @brief Registry managing all marker-object pairs
 * @details Demonstrates excellent extensibility architecture
 */
class MarkerRegistry {
private:
    std::map<std::string, MarkerObjectPair> registry;
    
public:
    /**
     * @brief Register a new marker-object pair
     */
    void registerPair(const std::string& name,
                     std::shared_ptr<MarkerDetector> marker,
                     std::shared_ptr<Object3D> object);
    
    /**
     * @brief Unregister a marker-object pair
     */
    void unregisterPair(const std::string& name);
    
    /**
     * @brief Enable/disable a specific pair
     */
    void setEnabled(const std::string& name, bool enabled);
    
    /**
     * @brief Get all registered pairs
     */
    const std::map<std::string, MarkerObjectPair>& getAllPairs() const { return registry; }
    
    /**
     * @brief Load registry from configuration file
     */
    bool loadFromConfig(const std::string& filename);
    
    /**
     * @brief Save registry to configuration file
     */
    bool saveToConfig(const std::string& filename) const;
    
    /**
     * @brief List all registered marker-object pairs
     */
    void printRegistry() const;
};

/**
 * @brief Factory for creating objects from configuration
 */
class ObjectFactory {
public:
    /**
     * @brief Create object from type name and parameters
     */
    static std::shared_ptr<Object3D> create(const std::string& type,
                                           const std::map<std::string, float>& params,
                                           const glm::vec3& color);
};

/**
 * @brief Factory for creating markers from configuration
 */
class MarkerFactory {
public:
    /**
     * @brief Create marker detector from type name and parameters
     */
    static std::shared_ptr<MarkerDetector> create(const std::string& type,
                                                  const std::map<std::string, float>& params);
};
