#ifndef MSE_GRAD_H
#define MSE_GRAD_H

#include "PlusML/gradient/loss_gradient.h"
#include "PlusML/util.h"

namespace plusml {
/**
 * \brief Class implementing Mean Square Error gradient
 */
class EXPORT MSEGrad : public LossGradient {
public:
  /**
   * \brief Calculate the MSE gradient for given inputs
   * \param w Parameters of the model
   * \param X Matrix of samples
   * \param y Matrix of targets
   * \return Matrix representing calculated gradient
   */
  Eigen::MatrixXf Compute(const Eigen::MatrixXf& w,
                          const Eigen::MatrixXf& X,
                          const Eigen::MatrixXf& y) const override;

  /**
   * \brief Set L2 regularization coefficient
   * \param c L2 regularization coefficient
   */
  void L2Regularization(const float c);
private:
  bool l2_regularization_enabled_ = false;
  float l2_regularization_coefficient_ = 0;
};
} //namespace plusml

#endif //MSE_GRAD_H
