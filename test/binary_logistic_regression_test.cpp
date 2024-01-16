#include "PlusML/binary_logistic_regression.h"

#include "testgen/testcase_gen.h"

#include <iostream>
#include <gtest/gtest.h>
#include <cmath>

TEST(BinaryLogisticRegressionTests, LinearlySeparableTest) {
  plusml::BinaryLogisticRegression model(2);

  float class_1_x = 0;
  float class_1_y = 0;

  float class_2_x = 25;
  float class_2_y = 25;

  float blob_radius = 10;

  uint64_t class_1_samples = 100;
  uint64_t class_2_samples = 100;

  const auto testcase = BinaryClassificationTestcase2d(
      class_1_x,
      class_1_y,
      0,
      class_2_x,
      class_2_y,
      1,
      blob_radius,
      class_1_samples,
      class_2_samples);

  model.FitSGD(testcase.first, testcase.second, 0.01, 10, 150);

  uint64_t right_predictions = 0;

  Eigen::MatrixXf predictions = model.Predict(testcase.first);

  for (uint64_t i = 0; i < testcase.first.rows(); ++i) {
    if (roundf(predictions(i, 0)) == testcase.second(i, 0)) {
      ++right_predictions;
    }
  }

  ASSERT_EQ(right_predictions, testcase.first.rows());
}

TEST(BinaryLogisticRegressionTests, NotLinearlySeparableTest) {
  plusml::BinaryLogisticRegression model(2);

  float class_1_x = 0;
  float class_1_y = 0;

  float class_2_x = 17;
  float class_2_y = 17;

  float blob_radius = 16;

  uint64_t class_1_samples = 100;
  uint64_t class_2_samples = 100;

  const auto testcase = BinaryClassificationTestcase2d(
      class_1_x,
      class_1_y,
      0,
      class_2_x,
      class_2_y,
      1,
      blob_radius,
      class_1_samples,
      class_2_samples);

  model.FitSGD(testcase.first, testcase.second, 0.01, 10, 150);

  uint64_t right_predictions = 0;

  Eigen::MatrixXf predictions = model.Predict(testcase.first);

  for (uint64_t i = 0; i < testcase.first.rows(); ++i) {
    if (roundf(predictions(i, 0)) == testcase.second(i, 0)) {
      ++right_predictions;
    }
  }

  ASSERT_TRUE((100.f * static_cast<float>(right_predictions) / static_cast<float>(testcase.first.rows())) > 70);
}