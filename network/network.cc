#include "network/network.h"

DEFINE_string(n_hidden, "100,50", "Number of neurons on hidden layers");
DEFINE_string(n_constants, "5,5,5",
              "Number of constant neurons on each non-output layer");

namespace {
std::vector<size_t> Split(const std::string &s_cpp) {
  std::vector<size_t> ret;
  const char *s = s_cpp.c_str();
  while (*s) {
    char *end;
    ret.push_back(strtoll(s, &end, 10));
    if (end == s) {
      fprintf(stderr, "Invalid string (no integer found): %s\n", s_cpp.c_str());
      exit(1);
    }
    s = end;
    if (*s != 0 && *s != ',') {
      fprintf(stderr,
              "Invalid string (number not terminated by comma or EOS): %s\n",
              s_cpp.c_str());
      exit(1);
    }
    if (*s == ',') {
      s++;
    }
  }
  return ret;
}
} // namespace

Network::Network(size_t n_input, size_t n_labels, const std::string &n_hidden,
                 const std::string &n_constants, NeuronFunction *function,
                 LossFunction *loss, NetworkTrainer *trainer)
    : n_input(n_input), n_labels(n_labels), function(function), loss(loss),
      trainer(trainer) {
  auto hidden = Split(n_hidden);
  auto constants_per_layer = Split(n_constants);
  if (hidden.size() + 1 != constants_per_layer.size()) {
    fprintf(stderr, "Invalid number of hidden layers vs constants!\n");
    exit(1);
  }
  layer_sizes.push_back(n_input);
  for (size_t h : hidden) {
    layer_sizes.push_back(h);
  }
  layer_sizes.push_back(n_labels);
  for (size_t c : constants_per_layer) {
    layer_constants.push_back(c);
  }
  // No constants on the output layer.
  layer_constants.push_back(0);
  size_t total_weights = 0;
  for (size_t i = 0; i + 1 < NumLayers(); i++) {
    total_weights += WeightsForLayer(i);
  }
  weights.resize(total_weights);
  weights.kind(TrainableVector::kWeights);
  trainer->InitVector(weights);
  size_t total_constants = 0;
  for (size_t num : layer_constants) {
    total_constants += num;
  }
  constants.resize(total_constants);
  weights.kind(TrainableVector::kConstants);
  trainer->InitVector(constants);
}

Network::Network(size_t n_input, size_t n_labels, NeuronFunction *function,
                 LossFunction *loss, NetworkTrainer *trainer)
    : Network(n_input, n_labels, FLAGS_n_hidden, FLAGS_n_constants, function,
              loss, trainer) {}
