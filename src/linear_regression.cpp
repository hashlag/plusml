#include "PlusML/linear_regression.h"

#include <forward_list>
#include <cassert>

namespace plusml {
LinearRegression::LinearRegression(uint64_t features, bool bias_enabled) {
  bias_enabled_ = bias_enabled;
  features_ = features + bias_enabled;

  w_ = Eigen::MatrixXf::Zero(features_, 1);
}

uint64_t LinearRegression::Features() const {
  return features_;
}

Eigen::MatrixXf LinearRegression::Parameters() const {
  return w_;
}

void LinearRegression::SetParameters(const Eigen::MatrixXf& parameters) {
  w_ = parameters;
}

void LinearRegression::FitSGD(const Eigen::MatrixXf& samples,
                              const Eigen::MatrixXf& targets,
                              const LossGradient& grad,
                              float learning_rate,
                              uint64_t batch_size,
                              uint64_t epochs) {
  assert(samples.rows() % batch_size == 0);

  const uint64_t batches_count = samples.rows() / batch_size;

  std::forward_list<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> batches;

  for (uint64_t batch_ix = 0; batch_ix < batches_count; ++batch_ix) {
    Eigen::MatrixXf SamplesBatch = samples.middleRows(batch_ix * batch_size, batch_size);
    Eigen::MatrixXf TargetsBatch = targets.middleRows(batch_ix * batch_size, batch_size);

    if (bias_enabled_) {
      SamplesBatch.conservativeResize(Eigen::NoChange, SamplesBatch.cols() + 1);
      SamplesBatch.col(SamplesBatch.cols() - 1) = Eigen::MatrixXf::Ones(SamplesBatch.rows(), 1);
    }

    batches.emplace_front(SamplesBatch, TargetsBatch);
  }

  for (uint64_t epoch = 0; epoch < epochs; ++epoch) {
    for (const auto& batch : batches) {
      Eigen::MatrixXf gradient = grad.Compute(w_, batch.first, batch.second);
      w_ -= learning_rate * gradient;
    }
  }
}

Eigen::MatrixXf LinearRegression::Predict(Eigen::MatrixXf samples) {
  if (bias_enabled_) {
    samples.conservativeResize(Eigen::NoChange, samples.cols() + 1);
    samples.col(samples.cols() - 1) = Eigen::MatrixXf::Ones(samples.rows(), 1);
  }

  return samples * w_;
}
} //namespace plusml
