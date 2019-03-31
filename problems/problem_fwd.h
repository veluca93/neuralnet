#pragma once
#include <gflags/gflags.h>
#include <memory>
#include <string>

DECLARE_string(problem);

// Interface for a classification problem.
class Problem {
public:
  static std::unique_ptr<Problem> New(bool train);
  static std::unique_ptr<Problem> New(const std::string &problem, bool train);
  virtual size_t NumInputs() const = 0;
  virtual size_t NumLabels() const = 0;
  virtual size_t NumExamples() const = 0;
  virtual std::string Name() const = 0;

  virtual void Example(size_t id, size_t *output, double *input) const = 0;

  virtual ~Problem() = default;
};
