#include "loss/loss.h"
#include "loss/softmax.h"
#include <gflags/gflags.h>

DEFINE_string(loss_function, "softmax", "Loss function to use.");

std::unique_ptr<LossFunction> LossFunction::New(const std::string &function) {
  if (function == SoftMax().Name()) {
    return std::make_unique<SoftMax>();
  }
  fprintf(stderr, "Invalid loss function name: %s\n",
          FLAGS_loss_function.c_str());
  exit(1);
}

std::unique_ptr<LossFunction> LossFunction::New() {
  return New(FLAGS_loss_function);
}
