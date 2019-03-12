#pragma once
#include "problems/problem.h"
#include <string>
#include <vector>

class MnistData : public Problem {
public:
  size_t InputSize() const override final { return 28 * 28; }
  size_t NumLabels() const override final { return 10; }
  size_t NumInputs() const override final { return vals.size(); }
  std::string Name() const override final { return "mnist"; }

  void Example(size_t id, size_t *out, double *in) const override final {
    *out = vals[id];
    for (size_t i = 0; i < InputSize(); i++) {
      in[i] = data[id * InputSize() + i] / 256.0;
    }
  }

  size_t NumExamples() const { return vals.size(); }
  static std::unique_ptr<MnistData> Load(bool train);

private:
  std::vector<uint8_t> vals;
  std::vector<uint8_t> data;
};
