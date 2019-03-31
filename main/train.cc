#include "init/init.h"
#include "network/network.h"
#include "neuron_function/neuron_function.h"
#include "problems/problem.h"
#include "training/training.h"
#include <algorithm>
#include <chrono>
#include <optional>
#include <random>
#include <signal.h>
#include <stdio.h>
#include <thread>

DEFINE_uint64(num_epochs, 1000, "Number of epochs for training");
DEFINE_double(validation_percentage, 0.1,
              "Percentage of items for the validation set");

volatile sig_atomic_t quit = 0;

void handleint(int sig) {
  if (quit)
    exit(1);
  quit = 1;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  auto train_data = Problem::New(/*train=*/true);
  auto test_data = Problem::New(/*train=*/false);
  auto neuron_function = NeuronFunction::New();
  auto loss_function = LossFunction::New();
  auto trainer = NetworkTrainer::New();
  auto initialzier = NetworkInitializer::New();
  Network network(train_data->NumInputs(), train_data->NumLabels(),
                  neuron_function.get(), loss_function.get(), trainer.get());
  network.Init(initialzier.get());
  std::vector<size_t> train_indices(train_data->NumInputs());
  std::iota(train_indices.begin(), train_indices.end(), 0);
  std::mt19937 rng;
  std::shuffle(train_indices.begin(), train_indices.end(), rng);
  size_t train_size =
      train_indices.size() * (1.0 - FLAGS_validation_percentage);
  std::vector<size_t> validation_indices(train_indices.begin() + train_size,
                                         train_indices.end());
  train_indices.resize(train_size);

  std::vector<size_t> test_indices(test_data->NumInputs());
  std::iota(test_indices.begin(), test_indices.end(), 0);

  std::optional<Network> best_network;
  std::optional<EpochStats> best_results;

  signal(SIGINT, handleint);

  for (size_t epoch = 0; epoch < FLAGS_num_epochs; epoch++) {
    auto start = std::chrono::high_resolution_clock::now();
    EpochStats train_stats =
        network.EvaluateAndTrain(train_data.get(), train_indices);
    EpochStats valid_stats =
        network.Evaluate(train_data.get(), validation_indices);
    if (!best_results.has_value() || best_results.value() < valid_stats) {
      best_results = valid_stats;
      best_network = network;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    fprintf(
        stderr,
        "\rEpoch %4lu: %6lums, train: %6.3f loss, %6.2f%% acc, valid: %6.3f "
        "loss, \033[37;1m%6.2f%%\033[;m acc\n",
        epoch,
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count(),
        train_stats.total_loss,
        100.0 * train_stats.num_correct / train_stats.epoch_size,
        valid_stats.total_loss,
        100.0 * valid_stats.num_correct / valid_stats.epoch_size);
    if (quit) {
      fprintf(stderr, "CTRL-C pressed, stopping training.\n");
      break;
    }
  }
  EpochStats test_stats = network.Evaluate(test_data.get(), test_indices);
  fprintf(stderr, "Testing results: %6.3f loss, \033[37;1m%6.2f%%\033[;m acc\n",
          test_stats.total_loss,
          100.0 * test_stats.num_correct / test_stats.epoch_size);
}
