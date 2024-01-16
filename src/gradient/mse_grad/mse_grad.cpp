#include "PlusML/gradient/mse_grad/mse_grad.h"

namespace plusml {
Eigen::MatrixXf MSEGrad::Compute(const Eigen::MatrixXf& w,
                          const Eigen::MatrixXf& X,
                          const Eigen::MatrixXf& y) const {

  Eigen::MatrixXf grad = 2 * (X.transpose() * ((X * w) - y)) / X.rows();

  if (l2_regularization_enabled_) {
    grad += (2 * l2_regularization_coefficient_) * w;
  }

  return grad;
}

void MSEGrad::L2Regularization(const float c) {
  l2_regularization_enabled_ = true;
  l2_regularization_coefficient_ = c;
}
} //namespace plusml