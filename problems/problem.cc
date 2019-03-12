#include "problems/problem.h"
#include "problems/mnist.h"

DEFINE_string(problem, "mnist", "Problem to load.");

std::unique_ptr<Problem> Problem::New(bool train) {
  if (FLAGS_problem == MnistData().Name()) {
    return MnistData::Load(train);
  }
  fprintf(stderr, "Invalid problem name: %s\n", FLAGS_problem.c_str());
  exit(1);
}
