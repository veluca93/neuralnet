#pragma once

#include "init/init_fwd.h"
#include <random>

class GaussianInitializer : public NetworkInitializer {
public:
  GaussianInitializer();
  void InitRegularWeights(double *w, size_t num, size_t layer, size_t src_size,
                          size_t dst_size) override final;
  void InitConstantsWeights(double *w, size_t num, size_t layer,
                            size_t src_size, size_t dst_size) override final;
  void InitConstants(double *c, size_t num, size_t layer) override final;

  std::string Name() const override final { return "gaussian"; }

private:
  std::mt19937 rng_;
};
