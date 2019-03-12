#include "neuron_function/neuron_function.h"
#include "neuron_function/relu.h"

DEFINE_string(neuron_function, "relu",
              "Function to be used to compute neuron output.");

std::unique_ptr<NeuronFunction> NeuronFunction::New() {
  if (FLAGS_neuron_function == Relu().Name()) {
    return std::make_unique<Relu>();
  }
  fprintf(stderr, "Invalid neuron function name: %s\n",
          FLAGS_neuron_function.c_str());
  exit(1);
}
