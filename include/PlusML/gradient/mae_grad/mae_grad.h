#ifndef MAE_GRAD_H
#define MAE_GRAD_H

#include "PlusML/gradient/loss_gradient.h"
#include "PlusML/util.h"

namespace plusml {
/**
 * \brief Class implementing Mean Absolute Error gradient
 */
class EXPORT MAEGrad : public LossGradient {
public:
  /**
   * \brief Calculate the MAE gradient for given inputs
   * \param w Parameters of the model
   * \param X Matrix of samples
   * \param y Matrix of targets
   * \return Matrix representing calculated gradient
   */
  Eigen::MatrixXf Compute(const Eigen::MatrixXf& w,
                          const Eigen::MatrixXf& X,
                          const Eigen::MatrixXf& y) const override;
};
} //namespace plusml

#endif //MAE_GRAD_H
