#include "neuron_function/relu.h"

double Relu::Output(const double *inputs, const double *weights,
                    size_t num_inputs, double *d_inputs,
                    double *d_weights) const {
  double result = 0;
  for (size_t i = 0; i < num_inputs; i++) {
    result += inputs[i] * weights[i];
  }
  if (result < 0)
    return 0;
  for (size_t i = 0; i < num_inputs; i++) {
    d_weights[i] = inputs[i];
    d_inputs[i] = weights[i];
  }
  return result;
}
