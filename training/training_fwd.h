#pragma once
#include <gflags/gflags.h>
#include <memory>
#include <string>
#include <vector>

DECLARE_string(network_trainer);

class TrainableVector {
public:
  enum Kind { kWeights = 0, kConstants = 1 };
  void resize(size_t sz) {
    values_.resize(sz);
    gradient_.resize(sz);
  }
  size_t size() const { return values_.size(); }
  const std::vector<double> &values() const { return values_; }
  std::vector<double> &values() { return values_; }
  const std::vector<double> &gradient() const { return gradient_; }
  std::vector<double> &gradient() { return gradient_; }

  size_t id() const { return id_; }
  void kind(Kind k) { kind_ = k; }
  Kind kind() { return kind_; }

private:
  std::vector<double> values_;
  std::vector<double> gradient_;
  static size_t next_id_;
  size_t id_ = next_id_++;
  Kind kind_;
};

class NetworkTrainer {
public:
  virtual void InitVector(const TrainableVector &vec) {}
  virtual void UpdateVector(TrainableVector *vec, double rate) = 0;
  void UpdateVector(TrainableVector *vec);
  virtual std::string Name() const = 0;
  virtual ~NetworkTrainer() = default;

  static std::unique_ptr<NetworkTrainer> New(const std::string &function);
  static std::unique_ptr<NetworkTrainer> New();
};
