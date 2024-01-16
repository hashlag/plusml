#include "PlusML/multiclass_logistic_regression.h"

namespace plusml {
MulticlassLogisticRegression::MulticlassLogisticRegression(uint64_t features, uint64_t classes, bool bias_enabled) {
  features_ = features;
  classes_ = classes;
  bias_enabled_ = bias_enabled;

  BinaryLogisticRegression classifier(features_, bias_enabled_);

  for (uint64_t class_ix = 0; class_ix < classes; ++class_ix) {
    classifiers_.emplace_front(class_ix, classifier);
  }
}

uint64_t MulticlassLogisticRegression::Features() const {
  return features_;
}

void MulticlassLogisticRegression::FitSGD(const Eigen::MatrixXf& samples,
                                          const Eigen::MatrixXi& targets,
                                          float learning_rate,
                                          uint64_t batch_size,
                                          uint64_t epochs) {
  for (auto& classifier : classifiers_) {
    uint64_t class_ix = std::get<0>(classifier);
    BinaryLogisticRegression& model = std::get<1>(classifier);

    Eigen::MatrixXf relabeled_targets = targets.unaryExpr([class_ix](const int x) {
        if (x == class_ix) {
          return 1.f;
        }

        return 0.f;
    });

    model.FitSGD(samples, relabeled_targets, learning_rate, batch_size, epochs);
  }
}

Eigen::MatrixXf MulticlassLogisticRegression::Predict(Eigen::MatrixXf samples) {
  Eigen::MatrixXf output(samples.rows(), classes_);

  for (uint64_t sample = 0; sample < samples.rows(); ++sample) {
    for (auto& classifier : classifiers_) {
      uint64_t class_ix = std::get<0>(classifier);
      BinaryLogisticRegression& model = std::get<1>(classifier);

      output(sample, class_ix) = model.Predict(samples.row(sample))(0, 0);
    }

    output.row(sample) = Softmax(output.row(sample));
  }

  return output;
}
} // plusml
