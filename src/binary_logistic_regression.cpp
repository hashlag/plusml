#include "PlusML/binary_logistic_regression.h"

#include <forward_list>
#include <cassert>

namespace plusml {
BinaryLogisticRegression::BinaryLogisticRegression(uint64_t features, bool bias_enabled) {
  bias_enabled_ = bias_enabled;
  features_ = features + bias_enabled;

  w_ = Eigen::MatrixXf::Zero(features_, 1);
}

uint64_t BinaryLogisticRegression::Features() const {
  return features_;
}

Eigen::MatrixXf BinaryLogisticRegression::Parameters() const {
  return w_;
}

void BinaryLogisticRegression::SetParameters(const Eigen::MatrixXf& parameters) {
  w_ = parameters;
}

void BinaryLogisticRegression::FitSGD(const Eigen::MatrixXf& samples,
                                      const Eigen::MatrixXf& targets,
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
      Eigen::MatrixXf gradient = Eigen::MatrixXf::Zero(1, features_);

      auto w_vector = Eigen::VectorXf{w_};

      for (uint64_t i = 0; i < batch_size; ++i) {
        gradient -= batch.first.row(i) * (batch.second(i, 0) - Sigmoid(w_vector.dot(batch.first.row(i))));
      }

      w_ -= learning_rate * gradient.transpose();
    }
  }
}

Eigen::MatrixXf BinaryLogisticRegression::Predict(Eigen::MatrixXf samples) {
  if (bias_enabled_) {
    samples.conservativeResize(Eigen::NoChange, samples.cols() + 1);
    samples.col(samples.cols() - 1) = Eigen::MatrixXf::Ones(samples.rows(), 1);
  }

  Eigen::MatrixXf result = (samples * w_).unaryExpr([](const float x) { return Sigmoid(x); });

  return result;
}
} //namespace plusml
