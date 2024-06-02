#include "PlusML/multiclass_svm.h"

#include <queue>
#include <tuple>
#include <cassert>
#include <iostream>
#include <algorithm>

namespace plusml {
MulticlassSVM::MulticlassSVM(uint64_t features, uint64_t classes, ClassificationMode mode, bool bias_enabled) {
  features_ = features;
  classes_ = classes;
  mode_ = mode;
  bias_enabled_ = bias_enabled;

  BinarySVM classifier(features_, bias_enabled_);

  if (mode_ == kOneVsAll) {
    for (uint64_t class_ix = 0; class_ix < classes; ++class_ix) {
      classifiers_.emplace_front(class_ix, 0, classifier);
    }
  }

  if (mode_ == kOneVsOne) {
    for (uint64_t class_ix = 0; class_ix < classes - 1; ++class_ix) {
      for (uint64_t against_class_ix = class_ix + 1; against_class_ix < classes; ++against_class_ix) {
        classifiers_.emplace_front(class_ix, against_class_ix, classifier);
      }
    }
  }
}

uint64_t MulticlassSVM::Features() const {
  return features_;
}

void MulticlassSVM::FitSGD(const Eigen::MatrixXf& samples,
                           const Eigen::MatrixXi& targets,
                           float learning_rate,
                           float l2_alpha,
                           uint64_t batch_size,
                           uint64_t epochs) {
  if (mode_ == kOneVsAll) {
    for (auto& classifier : classifiers_) {
      uint64_t class_ix = std::get<0>(classifier);
      BinarySVM& model = std::get<2>(classifier);

      Eigen::MatrixXf relabeled_targets = targets.unaryExpr([class_ix](const int x) {
        if (x == class_ix) {
          return 1.f;
        }

        return -1.f;
      });

      model.FitSGD(samples, relabeled_targets, learning_rate, l2_alpha, batch_size, epochs);
    }
  }

  if (mode_ == kOneVsOne) {
    std::vector<uint64_t> class_count(classes_, 0);

    for (uint64_t row = 0; row < targets.rows(); ++row) {
      ++class_count[targets(row, 0)];
    }

    for (auto& classifier : classifiers_) {
      uint64_t class_ix = std::get<0>(classifier);
      uint64_t against_class_ix = std::get<1>(classifier);
      BinarySVM& model = std::get<2>(classifier);

      Eigen::MatrixXf local_samples(class_count[class_ix] + class_count[against_class_ix], features_);
      Eigen::MatrixXf local_targets(class_count[class_ix] + class_count[against_class_ix], 1);

      uint64_t current_row = 0;

      for (uint64_t row = 0; row < targets.rows(); ++row) {
        if (targets(row, 0) == class_ix || targets(row, 0) == against_class_ix) {
          local_samples.row(current_row) = samples.row(row);

          if (targets(row, 0) == class_ix) {
            local_targets(current_row, 0) = 1.f;
          } else {
            local_targets(current_row, 0) = -1.f;
          }

          ++current_row;
        }
      }

      model.FitSGD(local_samples, local_targets, learning_rate, l2_alpha, batch_size, epochs);
    }
  }
}

Eigen::MatrixXi MulticlassSVM::Predict(Eigen::MatrixXf samples) {
  Eigen::MatrixXi output(samples.rows(), 1);

  if (mode_ == kOneVsAll) {
    std::queue<std::pair<float, uint64_t>> predictions;

    for (uint64_t sample = 0; sample < samples.rows(); ++sample) {
      predictions.emplace(std::numeric_limits<float>::min(), 0);

      for (auto& classifier : classifiers_) {
        const uint64_t class_ix = std::get<0>(classifier);
        BinarySVM& model = std::get<2>(classifier);

        float raw_prediction = model.Predict(samples.row(sample), false)(0, 0);

        if (raw_prediction > predictions.back().first) {
          predictions.back().first = raw_prediction;
          predictions.back().second = class_ix;
        }
      }
    }

    for (uint64_t sample = 0; sample < samples.rows(); ++sample) {
      output(sample, 0) = predictions.front().second;
      predictions.pop();
    }
  }

  if (mode_ == kOneVsOne) {
    for (uint64_t sample = 0; sample < samples.rows(); ++sample) {
      std::vector<uint64_t> votes(classes_, 0);

      for (auto& classifier : classifiers_) {
        uint64_t class_ix = std::get<0>(classifier);
        uint64_t against_class_ix = std::get<1>(classifier);
        BinarySVM& model = std::get<2>(classifier);

        float prediction = model.Predict(samples.row(sample), true)(0, 0);

        if (prediction == 1) {
          ++votes[class_ix];
        } else {
          ++votes[against_class_ix];
        }
      }

      auto predicted = std::max_element(votes.begin(), votes.end());
      output(sample, 0) = std::distance(votes.begin(), predicted);
    }
  }

  return output;
}
} // namespace plusml
