#include "PlusML/gradient/mae_grad/mae_grad.h"

namespace plusml {
Eigen::MatrixXf MAEGrad::Compute(const Eigen::MatrixXf& w,
                          const Eigen::MatrixXf& X,
                          const Eigen::MatrixXf& y) const {
  Eigen::MatrixXf grad = (X.transpose() * ((X * w) - y).unaryExpr([](float const n) { return Sign(n); })) / X.rows();
  return grad;
}
} //namespace plusml