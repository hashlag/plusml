#ifndef TESTCASE_GEN_H
#define TESTCASE_GEN_H

#include <utility>
#include <Eigen/Dense>

constexpr double kPi = 3.14159265358979323846;

float RandomFloatInRange(float x, float y);

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> LinearRegressionTestcase2d(float k, float b, float noise, uint64_t samples);
std::pair<Eigen::MatrixXf, Eigen::MatrixXf> BinaryClassificationTestcase2d(float c1_x,
                                                                           float c1_y,
                                                                           float c1_label,
                                                                           float c2_x,
                                                                           float c2_y,
                                                                           float c2_label,
                                                                           float base_radius,
                                                                           uint64_t c1_samples,
                                                                           uint64_t c2_samples);

std::pair<Eigen::MatrixXf, Eigen::MatrixXi> FourBlobsTestcase2d(std::array<uint64_t, 4> labels,
                                                                float base_radius,
                                                                float x_distance,
                                                                float y_distance,
                                                                uint64_t samples);

#endif //TESTCASE_GEN_H
