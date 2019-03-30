#pragma once
#include "loss/loss_fwd.h"

class SoftMax : public LossFunction {
public:
  std::pair<bool, double> Loss(const double *outputs, size_t num,
                               size_t correct,
                               double *d_outputs) const override final;
  std::string Name() const override final { return "softmax"; }
};
