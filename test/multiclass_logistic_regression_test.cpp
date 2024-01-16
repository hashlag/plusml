#include "PlusML/multiclass_logistic_regression.h"

#include "testgen/testcase_gen.h"

#include <iostream>
#include <gtest/gtest.h>

TEST(MulticlassLogisticRegressionTests, FourBlobsTest) {
  plusml::MulticlassLogisticRegression model(2, 4);

  const auto testcase = FourBlobsTestcase2d({0, 1, 2, 3}, 7, 30, 30, 100);

  model.FitSGD(testcase.first, testcase.second, 0.01, 10, 200);

  Eigen::MatrixXf predictions = model.Predict(testcase.first);

  uint64_t right_predictions = 0;

  for (uint64_t i = 0; i < testcase.second.rows(); ++i) {
    if (plusml::Argmax(predictions.row(i)) == testcase.second(i,  0)) {
      ++right_predictions;
    }
  }

  ASSERT_TRUE(100.f * static_cast<float>(right_predictions) / static_cast<float>(testcase.second.rows()) > 75);
}