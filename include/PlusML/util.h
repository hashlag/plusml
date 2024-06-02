#ifndef UTIL_H
#define UTIL_H

#define EXPORT __declspec(dllexport)
#include <Eigen/Dense>

namespace plusml {
EXPORT float Sign(float n);
EXPORT double Sign(double n);

EXPORT float Sigmoid(float n);
EXPORT double Sigmoid(double n);

EXPORT Eigen::VectorXf Softmax(const Eigen::VectorXf& vec);
EXPORT void SoftmaxInPlace(Eigen::VectorXf& vec);

EXPORT uint64_t Argmax(const Eigen::VectorXf& vec);
} //namespace plusml

#endif //UTIL_H
