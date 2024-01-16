#include "PlusML/multiclass_svm.h"

#include "testgen/testcase_gen.h"

#include <iostream>
#include <gtest/gtest.h>

TEST(MulticlassSVMTests, FourBlobsOneVsAllTest) {
  plusml::MulticlassSVM model(2, 4, plusml::kOneVsAll);

  const auto testcase = FourBlobsTestcase2d({0, 1, 2, 3}, 7, 30, 30, 100);

  model.FitSGD(testcase.first, testcase.second, 0.01, 0.01, 10, 200);

  Eigen::MatrixXi predictions = model.Predict(testcase.first);

  uint64_t right_predictions = 0;

  for (uint64_t i = 0; i < testcase.second.rows(); ++i) {
    if (predictions(i, 0) == testcase.second(i, 0)) {
      ++right_predictions;
    }
  }

  ASSERT_TRUE(100.f * static_cast<float>(right_predictions) / static_cast<float>(testcase.second.rows()) > 70);
}

TEST(MulticlassSVMTests, FourBlobsOneVsOneTest) {
  plusml::MulticlassSVM model(2, 4, plusml::kOneVsOne);

  const auto testcase = FourBlobsTestcase2d({0, 1, 2, 3}, 7, 30, 30, 100);

  model.FitSGD(testcase.first, testcase.second, 0.01, 0.01, 10, 200);

  Eigen::MatrixXi predictions = model.Predict(testcase.first);

  uint64_t right_predictions = 0;

  for (uint64_t i = 0; i < testcase.second.rows(); ++i) {
    if (predictions(i, 0) == testcase.second(i, 0)) {
      ++right_predictions;
    }
  }

  ASSERT_TRUE(100.f * static_cast<float>(right_predictions) / static_cast<float>(testcase.second.rows()) > 70);
}