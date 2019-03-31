#pragma once
#include "init/init.h"
#include "loss/loss.h"
#include "neuron_function/neuron_function.h"
#include "training/training.h"

struct EpochStats {
  size_t epoch_size;
  size_t num_correct;
  double total_loss;
};

class Network {
public:
  Network(size_t n_input, size_t n_labels, const std::string &n_hidden,
          const std::string &n_constants, NeuronFunction *function,
          LossFunction *loss, NetworkTrainer *trainer);
  Network(size_t n_input, size_t n_labels, NeuronFunction *function,
          LossFunction *loss, NetworkTrainer *trainer);
  void Init(NetworkInitializer *init);

private:
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
