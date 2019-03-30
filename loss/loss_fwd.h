#pragma once
#include <gflags/gflags.h>
#include <memory>
#include <stddef.h>

DECLARE_string(loss_function);

class LossFunction {
public:
  // Computes loss, correctness and output derivatives. `d_outputs` should be
  // zero-initialized.
  virtual std::pair<bool, double> Loss(const double *outputs, size_t num,
                                       size_t correct,
                                       double *d_outputs) const = 0;

  virtual std::string Name() const = 0;

  static std::unique_ptr<LossFunction> New(const std::string &function);
  static std::unique_ptr<LossFunction> New();

  ~LossFunction() = default;
};
