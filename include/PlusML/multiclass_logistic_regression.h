#ifndef MULTICLASS_LOGISTIC_REGRESSION_H
#define MULTICLASS_LOGISTIC_REGRESSION_H

#include "PlusML/util.h"

#include <cstdint>
#include <forward_list>
#include <Eigen/Dense>

#include "binary_logistic_regression.h"

namespace plusml {
/**
 * \brief Class implementing multiclassification logistic regression
 */
class EXPORT MulticlassLogisticRegression {
public:
  MulticlassLogisticRegression() = delete;

  /**
   * \brief Constructor for MulticlassLogisticRegression
   * \param features Number of features in each sample
   * \param classes Number of classes in classification problem
   * \param bias_enabled Specifies whether to use bias or not
   */
  MulticlassLogisticRegression(uint64_t features, uint64_t classes, bool bias_enabled = true);

  /**
   * \brief Get number of features in each sample for the model
   * \return Number of features in each sample for the model
   */
  uint64_t Features() const;

  /**
   * \brief Fit model using stochastic gradient descent with given hyperparameters
   * \param samples Matrix of samples (MxN where M is number of samples and N is number of features in each sample)
   * \param targets Matrix of targets (Mx1 where M is number of samples)
   * \param learning_rate Learning rate for SGD
   * \param batch_size Batch size for SGD
   * \param epochs Number of epochs for SGD
   */
  void FitSGD(const Eigen::MatrixXf& samples,
              const Eigen::MatrixXi& targets,
              float learning_rate,
              uint64_t batch_size,
              uint64_t epochs);

  /**
   * \brief Get predictions for a given matrix of samples
   * \param samples Matrix of samples (MxN where M is number of samples and N is number of features in each sample)
   * \return Matrix of predicted probabilities (MxN where M is number of samples and N is number of classes)
   */
  Eigen::MatrixXf Predict(Eigen::MatrixXf samples);

private:
  uint64_t features_;
  uint64_t classes_;

  bool bias_enabled_;

  std::forward_list<std::tuple<uint64_t, BinaryLogisticRegression>> classifiers_;
};

} //namespace plusml

#endif //MULTICLASS_LOGISTIC_REGRESSION_H
