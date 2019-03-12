#include "problems/problem.h"
#include <stdio.h>

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  auto train_data = Problem::New(/*train=*/true);
  auto test_data = Problem::New(/*train=*/false);
}
