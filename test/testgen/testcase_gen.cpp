#include "testcase_gen.h"
#include <cmath>
#include <random>
#include <algorithm>

float RandomFloatInRange(const float x, const float y) {
  return x + (rand() / (RAND_MAX / (y - x)));
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf>
LinearRegressionTestcase2d(float k, float b, float noise, uint64_t samples) {
  Eigen::MatrixXf X(samples, 1);
  Eigen::MatrixXf y(samples, 1);

  std::srand(std::time(0));

  for (uint64_t sample = 0; sample < samples; ++sample) {
    float x = RandomFloatInRange(-10, 10.1);
    X(sample, 0) = x + RandomFloatInRange(-noise, noise);
    y(sample, 0) = (k * x + b) + RandomFloatInRange(-noise, noise);
  }

  return std::make_pair(X, y);
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> BinaryClassificationTestcase2d(float c1_x,
                                                                           float c1_y,
                                                                           float c1_label,
                                                                           float c2_x,
                                                                           float c2_y,
                                                                           float c2_label,
                                                                           float base_radius,
                                                                           uint64_t c1_samples,
                                                                           uint64_t c2_samples) {
  Eigen::MatrixXf X(c1_samples + c2_samples, 2);
  Eigen::MatrixXf y(c1_samples + c2_samples, 1);

  uint64_t current_row = 0;

  std::srand(std::time(0));

  for (uint64_t sample = 0; sample < c1_samples; ++sample) {
    float alpha = 2 * kPi * RandomFloatInRange(0, 1);
    float radius = base_radius * std::sqrt(RandomFloatInRange(0, 1));
    float coord_x = radius * std::cos(alpha) + c1_x;
    float coord_y = radius * std::sin(alpha) + c1_y;

    X(current_row, 0) = coord_x;
    X(current_row, 1) = coord_y;

    y(current_row, 0) = c1_label;

    ++current_row;
  }

  for (uint64_t sample = 0; sample < c2_samples; ++sample) {
    float alpha = 2 * kPi * RandomFloatInRange(0, 1);
    float radius = base_radius * std::sqrt(RandomFloatInRange(0, 1));
    float coord_x = radius * std::cos(alpha) + c2_x;
    float coord_y = radius * std::sin(alpha) + c2_y;

    X(current_row, 0) = coord_x;
    X(current_row, 1) = coord_y;

    y(current_row, 0) = c2_label;

    ++current_row;
  }

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> p_matrix(X.rows());
  p_matrix.setIdentity();

  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
  std::mt19937 engine(seed);

  std::shuffle(p_matrix.indices().data(), p_matrix.indices().data() + p_matrix.indices().size(), engine);

  Eigen::MatrixXf X_shuffled = p_matrix * X;
  Eigen::MatrixXf y_shuffled = p_matrix * y;

  return std::make_pair(X_shuffled, y_shuffled);
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXi> FourBlobsTestcase2d(std::array<uint64_t, 4> labels,
                                                                float base_radius,
                                                                float x_distance,
                                                                float y_distance,
                                                                uint64_t samples) {

  Eigen::MatrixXf X(samples * 4, 2);
  Eigen::MatrixXi y(samples * 4, 1);

  std::srand(std::time(0));

  std::array<std::pair<float, float>, 4> base_points {
    {
      std::make_pair(0.f, 0.f),
      std::make_pair(0.f, y_distance),
      std::make_pair(x_distance, 0.f),
      std::make_pair(x_distance, y_distance)
    }};

  uint64_t current_row = 0;
  auto label_iter = labels.begin();

  for (auto point : base_points) {
    for (uint64_t i = 0; i < samples; ++i) {
      float alpha = 2 * kPi * RandomFloatInRange(0, 1);
      float radius = base_radius * std::sqrt(RandomFloatInRange(0, 1));
      float coord_x = radius * std::cos(alpha) + point.first;
      float coord_y = radius * std::sin(alpha) + point.second;

      X(current_row, 0) = coord_x;
      X(current_row, 1) = coord_y;

      y(current_row, 0) = *label_iter;

      ++current_row;
    }
    ++label_iter;
  }

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> p_matrix(X.rows());
  p_matrix.setIdentity();

  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
  std::mt19937 engine(seed);

  std::shuffle(p_matrix.indices().data(), p_matrix.indices().data() + p_matrix.indices().size(), engine);

  Eigen::MatrixXf X_shuffled = p_matrix * X;
  Eigen::MatrixXi y_shuffled = p_matrix * y;

  return std::make_pair(X_shuffled, y_shuffled);
}
