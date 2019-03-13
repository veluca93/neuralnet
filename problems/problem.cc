#include "problems/problem.h"
#include "problems/mnist.h"

DEFINE_string(problem, "mnist", "Problem to load.");

std::unique_ptr<Problem> Problem::New(const std::string &problem, bool train) {
  if (problem == MnistData().Name()) {
    return MnistData::Load(train);
  }
  fprintf(stderr, "Invalid problem name: %s\n", FLAGS_problem.c_str());
  exit(1);
}

std::unique_ptr<Problem> Problem::New(bool train) {
  return New(FLAGS_problem, train);
}
