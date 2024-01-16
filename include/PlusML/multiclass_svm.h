#ifndef MULTICLASS_SVM_H
#define MULTICLASS_SVM_H

#include <forward_list>

#include "PlusML/util.h"
#include <Eigen/Dense>

#include "PlusML/binary_svm.h"

namespace plusml {
enum ClassificationMode {
  kOneVsAll,
  kOneVsOne
};

/**
 * \brief Class implementing multiclassification SVM
 *
 * Pass `plusml::kOneVsOne` or `plusml::kOneVsAll` to the constructor to select classification mode.
 */
class EXPORT MulticlassSVM {
public:
  MulticlassSVM() = delete;

  /**
   * \brief Constructor for MulticlassSVM
   * \param features Number of features in each sample
   * \param classes Number of classes in classification problem
   * \param mode Classification mode (`plusml::kOneVsOne` or `plusml::kOneVsAll`)
   * \param bias_enabled Specifies whether to use bias or not
   */
  MulticlassSVM(uint64_t features, uint64_t classes, ClassificationMode mode, bool bias_enabled = true);

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
   * \param l2_alpha Coefficient for L2 regularization
   * \param batch_size Batch size for SGD
   * \param epochs Number of epochs for SGD
   */
  void FitSGD(const Eigen::MatrixXf& samples,
              const Eigen::MatrixXi& targets,
              float learning_rate,
              float l2_alpha,
              uint64_t batch_size,
              uint64_t epochs);

  /**
   * \brief Get predictions for a given matrix of samples
   * \param samples Matrix of samples (MxN where M is number of samples and N is number of features in each sample)
   * \return Matrix of predicted classes (Mx1 where M is number of samples)
   */
  Eigen::MatrixXi Predict(Eigen::MatrixXf samples);

private:
  ClassificationMode mode_;

  uint64_t features_;
  uint64_t classes_;

  bool bias_enabled_;

  std::forward_list<std::tuple<uint64_t, uint64_t, BinarySVM>> classifiers_;
};
} // plusml

#endif //MULTICLASS_SVM_H
