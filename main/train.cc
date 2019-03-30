#include "network/network.h"
#include "neuron_function/neuron_function.h"
#include "problems/problem.h"
#include "training/training.h"
#include <stdio.h>

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  auto train_data = Problem::New(/*train=*/true);
  auto test_data = Problem::New(/*train=*/false);
  auto neuron_function = NeuronFunction::New();
  auto loss_function = LossFunction::New();
  auto trainer = NetworkTrainer::New();
  Network network(train_data->NumInputs(), train_data->NumLabels(),
                  neuron_function.get(), loss_function.get(), trainer.get());
}
