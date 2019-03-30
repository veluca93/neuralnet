#include "loss/softmax.h"
#include <algorithm>
#include <math.h>
#include <numeric>

std::pair<bool, double> SoftMax::Loss(const double *outputs, size_t num,
                                      size_t correct, double *d_outputs) const {
  size_t max_pos = std::max_element(outputs, outputs + num) - outputs;
  double max = *std::max_element(outputs, outputs + num);
  for (size_t i = 0; i < num; i++) {
    d_outputs[i] = std::exp(outputs[i] - max);
  }
  double sum = std::accumulate(d_outputs, d_outputs + num, 0);
  for (size_t i = 0; i < num; i++) {
    d_outputs[i] /= sum;
  }
  double loss = -std::log(d_outputs[correct] + 1e-200);
  for (size_t i = 0; i < num; i++) {
    if (i == correct) {
      d_outputs[i] -= 1;
    } else {
      d_outputs[i] = 0;
    }
  }
  return {max_pos == correct, loss};
}
