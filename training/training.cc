#include "training/training.h"
#include "training/sgd.h"

DEFINE_string(network_trainer, "sgd", "Network training method.");
DEFINE_double(learn_rate_weights, 0.1, "Weight learning rate.");
DEFINE_double(learn_rate_constants, 0.01, "Constants learning rate");

size_t TrainableVector::next_id_ = 0;

void NetworkTrainer::UpdateVector(TrainableVector *vec) {
  switch (vec->kind()) {
  case TrainableVector::kConstants:
    return UpdateVector(vec, FLAGS_learn_rate_constants);
  case TrainableVector::kWeights:
    return UpdateVector(vec, FLAGS_learn_rate_weights);
  }
}

std::unique_ptr<NetworkTrainer>
NetworkTrainer::New(const std::string &function) {
  if (function == SGD().Name()) {
    return std::make_unique<SGD>();
  }
  fprintf(stderr, "Invalid loss function name: %s\n",
          FLAGS_network_trainer.c_str());
  exit(1);
}

std::unique_ptr<NetworkTrainer> NetworkTrainer::New() {
  return New(FLAGS_network_trainer);
}
