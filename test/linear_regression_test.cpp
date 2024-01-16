#include "PlusML/linear_regression.h"
#include "PlusML/gradient.h"
#include "testgen/testcase_gen.h"

#include <iostream>
#include <random>

#include <gtest/gtest.h>

TEST(LinearRegressionTests, LinearRegressionMSE) {
  float k = 7;
  float b = 100;

  plusml::LinearRegression model(1, true);
  const auto testcase = LinearRegressionTestcase2d(k, b, 1, 100);

  model.FitSGD(testcase.first, testcase.second, plusml::MSEGrad(), 0.01, 5, 200);

  ASSERT_NEAR(model.Parameters()(0, 0), k, 1);
  ASSERT_NEAR(model.Parameters()(1, 0), b, 2);

  float x = 10;

  Eigen::MatrixXf samples(1, 1);
  samples << x;

  ASSERT_NEAR(model.Predict(samples)(0, 0), k*x + b, 7);
}

TEST(LinearRegressionTests, LinearRegressionMAE) {
  float k = 7;
  float b = 100;

  plusml::LinearRegression model(1, true);
  const auto testcase = LinearRegressionTestcase2d(k, b, 1, 100);

  model.FitSGD(testcase.first, testcase.second, plusml::MAEGrad(), 0.05, 5, 4000);

  ASSERT_NEAR(model.Parameters()(0, 0), k, 1);
  ASSERT_NEAR(model.Parameters()(1, 0), b, 2);

  float x = 10;

  Eigen::MatrixXf samples(1, 1);
  samples << x;

  ASSERT_NEAR(model.Predict(samples)(0, 0), k*x + b, 7);
}
