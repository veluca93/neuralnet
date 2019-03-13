#pragma once
#include "neuron_function/neuron_function.h"

class Network {
public:
  Network(size_t n_input, size_t n_labels, const std::string &n_hidden,
          const std::string &n_constants, NeuronFunction *function);
  Network(size_t n_input, size_t n_labels, NeuronFunction *function);

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
  std::vector<double> weights;
  std::vector<double> constants;
  std::vector<size_t> layer_sizes;
  std::vector<size_t> layer_constants;
};
