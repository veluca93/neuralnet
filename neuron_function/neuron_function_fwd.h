#pragma once
#include <gflags/gflags.h>
#include <memory>
#include <stddef.h>

DECLARE_string(neuron_function);

class NeuronFunction {
public:
  // Allows applying arbitrary transformations to the input values.
  virtual void TransformInput(double *inputs, size_t num) const {}

  // Compute output and input derivatives of a neuron. d_inputs and d_weights
  // are 0-initialized.
  virtual double Output(const double *inputs, const double *weights,
                        size_t num_inputs, double *d_inputs,
                        double *d_weights) const = 0;

  // Allows applying arbitrary transofrmations to the output values and
  // derivatives.
  virtual void TransformOutput(double *outputs, size_t num) const {}
  virtual void TransformOutputDerivative(double *d_outputs, size_t num) const {}

  virtual std::string Name() const = 0;

  static std::unique_ptr<NeuronFunction> New(const std::string &function);
  static std::unique_ptr<NeuronFunction> New();

  ~NeuronFunction() = default;
};
