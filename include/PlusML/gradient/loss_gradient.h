#ifndef LOSS_GRADIENT_H
#define LOSS_GRADIENT_H

#include "PlusML/util.h"
#include <Eigen/Dense>

namespace plusml {
/**
 * \brief Base class for loss gradient implementations
 */
class EXPORT LossGradient {
public:
  /**
   * \brief Virtual function describing the gradient computation function interface
   * \param parameters Parameters of the model
   * \param X Matrix of samples
   * \param y Matrix of targets
   * \return Matrix representing calculated gradient
   */
  virtual Eigen::MatrixXf Compute(const Eigen::MatrixXf& parameters,
                                  const Eigen::MatrixXf& X,
                                  const Eigen::MatrixXf& y) const = 0;

  virtual ~LossGradient() = default;
};
} //namespace plusml

#endif //LOSS_GRADIENT_H
