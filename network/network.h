#pragma once
#include "init/init.h"
#include "loss/loss.h"
#include "neuron_function/neuron_function.h"
#include "problems/problem.h"
#include "training/training.h"
#include <assert.h>

struct EpochStats {
  size_t epoch_size;
  size_t num_correct;
  double total_loss;
  bool operator<(const EpochStats &other) {
    assert(other.epoch_size == epoch_size);
    if (num_correct < other.num_correct)
      return true;
    if (num_correct > other.num_correct)
      return false;
    return other.total_loss < total_loss;
  }
};

class Network {
public:
  Network(size_t n_input, size_t n_labels, const std::string &n_hidden,
          const std::string &n_constants, NeuronFunction *function,
          LossFunction *loss, NetworkTrainer *trainer);
  Network(size_t n_input, size_t n_labels, NeuronFunction *function,
          LossFunction *loss, NetworkTrainer *trainer);
  void Init(NetworkInitializer *init);
  EpochStats Evaluate(Problem *problem, const std::vector<size_t> &indices) {
    return Evaluate(problem, indices, /*train=*/false);
  }
  EpochStats EvaluateAndTrain(Problem *problem,
                              const std::vector<size_t> &indices) {
    return Evaluate(problem, indices, /*train=*/true);
  }

private:
  EpochStats Evaluate(Problem *problem, const std::vector<size_t> &indices,
                      bool train);

  size_t WeightsForLayer(size_t i) const {
    // Layer i has (layer_sizes[i]+layer_constants[i]) * layer_sizes[i+1]
    // weights.
    return (layer_sizes[i] + layer_constants[i]) * layer_sizes[i + 1];
  }
  size_t NumLayers() const { return layer_sizes.size(); }
  size_t n_input;
  size_t n_labels;
  NeuronFunction *function;
  LossFunction *loss;
  NetworkTrainer *trainer;
  TrainableVector weights;
  TrainableVector constants;
  std::vector<size_t> layer_sizes;
  std::vector<size_t> layer_constants;
};
