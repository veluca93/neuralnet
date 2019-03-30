#include "training/sgd.h"

void SGD::UpdateVector(TrainableVector *vec, double rate) {
  for (size_t i = 0; i < vec->size(); i++) {
    vec->values()[i] -= rate * vec->gradient()[i];
    vec->gradient()[i] = 0.0;
  }
}
