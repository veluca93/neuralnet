#include "neuron_function/neuron_function.h"

class Relu : public NeuronFunction {
public:
  std::string Name() const override final { return "relu"; }
  double Output(const double *inputs, const double *weights, size_t num_inputs,
                double *d_inputs, double *d_weights) const override final;
};
