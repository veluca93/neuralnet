#pragma once
#include "training/training_fwd.h"

class SGD : public NetworkTrainer {
public:
  virtual void UpdateVector(TrainableVector *vec, double rate) override final;
  virtual std::string Name() const override final { return "sgd"; }
};
