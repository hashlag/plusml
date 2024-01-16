#include "PlusML/util.h"
#include <cmath>

namespace plusml {
float Sign(const float n) {
  if (n > 0) {
    return 1;
  }

  if (n < 0) {
    return -1;
  }

  return 0;
}

double Sign(const double n) {
  if (n > 0) {
    return 1;
  }

  if (n < 0) {
    return -1;
  }

  return 0;
}

float Sigmoid(const float n) {
  return 1.f / (1.f + std::exp(-n));
}

double Sigmoid(const double n) {
  return 1.f / (1.f + std::exp(-n));
}

uint64_t Argmax(const Eigen::VectorXf& vec) {
  float max_value = std::numeric_limits<float>::min();
  uint64_t index = 0;

  for (uint64_t ix = 0; ix < vec.size(); ++ix) {
    if (vec(ix) > max_value) {
      max_value = vec(ix);
      index = ix;
    }
  }

  return index;
}

Eigen::VectorXf Softmax(const Eigen::VectorXf& vec) {
  float denominator = 0;

  for (const auto elem : vec) {
    denominator += std::exp(elem);
  }

  Eigen::VectorXf output = vec.unaryExpr([denominator](const float x){return std::exp(x) / denominator;});

  return output;
}

void SoftmaxInPlace(Eigen::VectorXf& vec) {
  float denominator = 0;

  for (const auto elem : vec) {
    denominator += std::exp(elem);
  }

  vec = vec.unaryExpr([denominator](const float x){return std::exp(x) / denominator;});
}
} //namespace plusml