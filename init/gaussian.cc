#include "init/gaussian.h"
#include <gflags/gflags.h>

DEFINE_uint64(gaussian_seed, 0, "Seed for gaussian initialization");

DEFINE_double(gaussian_regular_weights_mean, 0.0,
              "Mean of regular weights for gaussian initialization");
DEFINE_double(
    gaussian_regular_weights_sigma_mul, 1.0,
    "Multiplier for sigma of regular weights for gaussian initialization");

DEFINE_double(gaussian_constants_weights_mean, 0.0,
              "Mean of constants weights for gaussian initialization");
DEFINE_double(
    gaussian_constants_weights_sigma_mul, 1.0,
    "Multiplier for sigma of constants weights for gaussian initialization");

DEFINE_double(gaussian_constants_mean, 0.5,
              "Mean of constants for gaussian initialization");
DEFINE_double(gaussian_constants_sigma, 0.5,
              "Sigma of constants for gaussian initialization");

GaussianInitializer::GaussianInitializer() : rng_(FLAGS_gaussian_seed) {}

void GaussianInitializer::InitRegularWeights(double *w, size_t num,
                                             size_t layer, size_t src_size,
                                             size_t dst_size) {
  double sigma = FLAGS_gaussian_regular_weights_sigma_mul *
                 std::sqrt(2.0 / (src_size + dst_size));
  std::normal_distribution<double> dist(FLAGS_gaussian_regular_weights_mean,
                                        sigma);
  for (size_t i = 0; i < num; i++) {
    w[i] = dist(rng_);
  }
}

void GaussianInitializer::InitConstantsWeights(double *w, size_t num,
                                               size_t layer, size_t src_size,
                                               size_t dst_size) {
  double sigma = FLAGS_gaussian_constants_weights_sigma_mul *
                 std::sqrt(2.0 / (src_size + dst_size));
  std::normal_distribution<double> dist(FLAGS_gaussian_constants_weights_mean,
                                        sigma);
  for (size_t i = 0; i < num; i++) {
    w[i] = dist(rng_);
  }
}

void GaussianInitializer::InitConstants(double *c, size_t num, size_t layer) {
  std::normal_distribution<double> dist(FLAGS_gaussian_constants_mean,
                                        FLAGS_gaussian_constants_sigma);
  for (size_t i = 0; i < num; i++) {
    c[i] = dist(rng_);
  }
}
