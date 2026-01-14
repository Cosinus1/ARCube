/**
 * @file Exceptions.hpp
 * @brief Custom exception types for AR application
 * @author Olivier Deruelle
 * @date 2025
 */

#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>

/**
 * @brief Base exception class for AR application
 */
class ARException : public std::runtime_error {
public:
    explicit ARException(const std::string& m) : std::runtime_error(m) {}
};

/**
 * @brief Exception for validation errors
 */
class ValidationException : public ARException {
public:
    explicit ValidationException(const std::string& m) 
        : ARException("Validation error: " + m) {}
};

/**
 * @brief Exception for resource allocation/management errors
 */
class ResourceException : public ARException {
public:
    explicit ResourceException(const std::string& m) 
        : ARException("Resource error: " + m) {}
};

/**
 * @brief Exception for processing/computation errors
 */
class ProcessingException : public ARException {
public:
    explicit ProcessingException(const std::string& m) 
        : ARException("Processing error: " + m) {}
};

/**
 * @brief Helper: throws ValidationException if condition fails
 * @param cond Condition to validate
 * @param msg Error message if condition is false
 */
inline void require(bool cond, const std::string& msg) {
    if (!cond) throw ValidationException(msg);
}

/**
 * @brief Helper: Convert cv::Exception to ProcessingException
 * @param e OpenCV exception
 * @param ctx Context description
 */
inline void rethrowCv(const cv::Exception& e, const std::string& ctx) {
    std::ostringstream oss;
    oss << ctx << " | cv::Exception: (" << e.code << ") " << e.what();
    throw ProcessingException(oss.str());
}
