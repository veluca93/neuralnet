#include "network/network.h"
#include <numeric>

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
  constants.kind(TrainableVector::kConstants);
  trainer->InitVector(constants);
}

Network::Network(size_t n_input, size_t n_labels, NeuronFunction *function,
                 LossFunction *loss, NetworkTrainer *trainer)
    : Network(n_input, n_labels, FLAGS_n_hidden, FLAGS_n_constants, function,
              loss, trainer) {}

void Network::Init(NetworkInitializer *init) {
  size_t wstart = 0;
  size_t cstart = 0;
  for (size_t l = 0; l + 1 < NumLayers(); l++) {
    for (size_t n = 0; n < layer_sizes[l + 1]; n++) {
      init->InitRegularWeights(weights.values().data() + wstart, layer_sizes[l],
                               l, layer_sizes[l] + layer_constants[l],
                               layer_sizes[l + 1]);
      init->InitConstantsWeights(
          weights.values().data() + wstart + layer_sizes[l], layer_constants[l],
          l, layer_sizes[l] + layer_constants[l], layer_sizes[l + 1]);
      wstart += layer_sizes[l] + layer_constants[l];
    }
    init->InitConstants(constants.values().data() + cstart, layer_constants[l],
                        l);
    cstart += layer_constants[l];
  }
}

void Network::RunNetwork(double *inputs, double *d_inputs, double *d_weights) {
  size_t wstart = 0;
  size_t istart = 0;
  for (size_t l = 0; l + 1 < NumLayers(); l++) {
    size_t lsize = layer_sizes[l] + layer_constants[l];
    size_t nistart = istart + lsize;
    for (size_t i = 0; i < layer_sizes[l + 1]; i++) {
      inputs[nistart + i] =
          function->Output(inputs + istart, weights.values().data() + wstart,
                           lsize, d_inputs + wstart, d_weights + wstart);
      wstart += lsize;
    }
    istart = nistart;
  }
}

void Network::UpdateGradients(const double *d_inputs, const double *d_weights,
                              const double *d_outputs, double *grad_inputs) {
  size_t num_constants =
      std::accumulate(layer_constants.begin(), layer_constants.end(), 0);
  size_t num_neurons =
      std::accumulate(layer_sizes.begin(), layer_sizes.end(), num_constants);

  size_t istart = num_neurons - n_labels;
  for (size_t i = 0; i < n_labels; i++) {
    grad_inputs[istart + i] = d_outputs[i];
  }
  size_t wstart = weights.size();
  size_t cstart = num_constants;
  for (size_t l = NumLayers() - 1; l > 0; l--) {
    size_t nistart = istart;
    istart -= layer_sizes[l - 1] + layer_constants[l - 1];
    for (size_t i = 0; i < layer_sizes[l]; i++) {
      wstart -= layer_constants[l - 1] + layer_sizes[l - 1];
      for (size_t j = 0; j < layer_constants[l - 1] + layer_sizes[l - 1]; j++) {
        grad_inputs[istart + j] +=
            d_inputs[wstart + j] * grad_inputs[nistart + i];
        weights.gradient()[istart + i] =
            d_weights[wstart + j] * grad_inputs[nistart + i];
      }
    }
    cstart -= layer_constants[l - 1];
    for (size_t i = 0; i < layer_constants[l - 1]; i++) {
      constants.gradient()[cstart + i] =
          grad_inputs[istart + layer_sizes[l - 1] + i];
    }
  }
}

DEFINE_uint64(batch_size, 100, "Batch size");

EpochStats Network::Evaluate(Problem *problem,
                             const std::vector<size_t> &indices, bool train) {
  size_t num_neurons =
      std::accumulate(layer_sizes.begin(), layer_sizes.end(), 0);
  num_neurons +=
      std::accumulate(layer_constants.begin(), layer_constants.end(), 0);

  std::vector<double> inputs(num_neurons);

  std::vector<double> d_weights(weights.size());
  std::vector<double> d_inputs(weights.size());
  std::vector<double> d_output(n_labels);
  std::vector<double> grad_inputs(num_neurons);

  EpochStats ret{};
  for (size_t batch_start = 0; batch_start < indices.size();
       batch_start += FLAGS_batch_size) {
    // TODO: parallelize this loop.
    std::fill(d_weights.begin(), d_weights.end(), 0);
    std::fill(d_inputs.begin(), d_inputs.end(), 0);
    std::fill(grad_inputs.begin(), grad_inputs.end(), 0);
    for (size_t ex = batch_start;
         ex < indices.size() && ex < batch_start + FLAGS_batch_size; ex++) {
      size_t idx = indices[ex];
      fprintf(stderr, "%10lu done\r", ret.epoch_size);
      ret.epoch_size++;
      size_t correct;
      problem->Example(idx, &correct, inputs.data());
      function->TransformInput(inputs.data(), n_input);
      size_t const_istart = layer_sizes[0];
      size_t const_start = 0;
      for (size_t i = 0; i + 1 < NumLayers(); i++) {
        for (size_t c = 0; c < layer_constants[i]; c++) {
          inputs[const_istart + c] = constants.values()[const_start + c];
        }
        const_istart += layer_constants[i] + layer_sizes[i + 1];
        const_start += layer_constants[i];
      }
      RunNetwork(inputs.data(), d_inputs.data(), d_weights.data());
      function->TransformOutput(inputs.data() + inputs.size() - n_labels,
                                n_labels);
      auto res = loss->Loss(inputs.data() + inputs.size() - n_labels, n_labels,
                            correct, d_output.data());
      if (res.first) {
        ret.num_correct++;
      }
      ret.total_loss += res.second;
      if (train) {
        function->TransformOutputDerivative(d_output.data(), n_labels);
        UpdateGradients(d_inputs.data(), d_weights.data(), d_output.data(),
                        grad_inputs.data());
      }
    }
    if (train) {
      trainer->UpdateVector(&weights);
      trainer->UpdateVector(&constants);
    }
  }
  return ret;
}
