cmake_minimum_required(VERSION 3.20)
project(plusml)

set(CMAKE_CXX_STANDARD 20)

set(SOURCES
        include/PlusML/linear_regression.h
        src/linear_regression.cpp
        include/PlusML/util.h
        include/PlusML/gradient/loss_gradient.h
        include/PlusML/gradient/mse_grad/mse_grad.h
        include/PlusML/gradient/mae_grad/mae_grad.h
        src/util.cpp
        include/PlusML/gradient.h
        src/gradient/mse_grad/mse_grad.cpp
        src/gradient/mae_grad/mae_grad.cpp
        src/binary_svm.cpp
        include/PlusML/binary_svm.h
        include/PlusML/binary_logistic_regression.h
        src/binary_logistic_regression.cpp
        src/multiclass_svm.cpp
        include/PlusML/multiclass_svm.h
        src/multiclass_logistic_regression.cpp
        include/PlusML/multiclass_logistic_regression.h
)

add_library(plusml SHARED ${SOURCES})

target_include_directories(plusml PUBLIC include third-party/eigen)

# Uncomment the TESTING_SECTION to enable tests
# TESTING_SECTION : begin

#include(FetchContent)
#FetchContent_Declare(
#        googletest
#        GIT_REPOSITORY https://github.com/google/googletest.git
#        GIT_TAG release-1.12.1
#)
#
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(googletest)
#
#enable_testing()
#
#add_executable(test_plusml
#        test/linear_regression_test.cpp
#        test/testgen/testcase_gen.cpp
#        test/testgen/testcase_gen.h
#        test/binary_svm_test.cpp
#        test/binary_logistic_regression_test.cpp
#        test/multiclass_svm_test.cpp
#        test/multiclass_logistic_regression_test.cpp
#)
#
#target_link_libraries(test_plusml plusml gtest gtest_main)
#
#include(GoogleTest)
#gtest_discover_tests(test_plusml)

# TESTING_SECTION : end