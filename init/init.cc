#include "init/init.h"
#include "init/gaussian.h"
#include <gflags/gflags.h>

DEFINE_string(network_initializer, "gaussian",
              "Network initialization method.");

std::unique_ptr<NetworkInitializer>
NetworkInitializer::New(const std::string &function) {
  if (function == GaussianInitializer().Name()) {
    return std::make_unique<GaussianInitializer>();
  }
  fprintf(stderr, "Invalid loss function name: %s\n",
          FLAGS_network_initializer.c_str());
  exit(1);
}

std::unique_ptr<NetworkInitializer> NetworkInitializer::New() {
  return New(FLAGS_network_initializer);
}
