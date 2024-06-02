#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "PlusML/gradient.h"
#include "PlusML/util.h"

#include <cstdint>
#include <Eigen/Dense>

namespace plusml {
/**
 * \brief Basic linear regression class
 */
class EXPORT LinearRegression {
public:
  LinearRegression() = delete;

  /**
   * \brief Constructor for LinearRegression class
   * \param features Number of features in each sample
   * \param bias_enabled Specifies whether to use bias or not
   *
   * Initializes model's parameters to zero
   */
  explicit LinearRegression(uint64_t features, bool bias_enabled = true);

  /**
   * \brief Get number of features in each sample for the model
   * \return Number of features in each sample for the model
   */
  uint64_t Features() const;

  /**
   * \brief Get current parameters of the model
   * \return Parameters matrix
   */
  Eigen::MatrixXf Parameters() const;

  /**
   * \brief Set desired parameters for the model
   * \param parameters Matrix representing parameters to set
   */
  void SetParameters(const Eigen::MatrixXf& parameters);

  /**
   * \brief Fit model using stochastic gradient descent with given hyperparameters
   * \param samples Matrix of samples (MxN where M is number of samples and N is number of features in each sample)
   * \param targets Matrix of targets (Mx1 where M is number of samples)
   * \param grad Class representing a gradient calculation technique (a derived from the LossGradient base class)
   * \param learning_rate Learning rate for SGD
   * \param batch_size Batch size for SGD
   * \param epochs Number of epochs for SGD
   */
  void FitSGD(const Eigen::MatrixXf& samples,
              const Eigen::MatrixXf& targets,
              const LossGradient& grad,
              float learning_rate,
              uint64_t batch_size,
              uint64_t epochs);

  /**
   * \brief Get predictions for a given matrix of samples
   * \param samples Matrix of samples (MxN where M is number of samples and N is number of features in each sample)
   * \return Matrix of predicted values (Mx1 where M is number of samples)
   */
  Eigen::MatrixXf Predict(Eigen::MatrixXf samples);

private:
  uint64_t features_;
  bool bias_enabled_;
  Eigen::MatrixXf w_;
};
} //namespace plusml

#endif //LINEAR_REGRESSION_H
