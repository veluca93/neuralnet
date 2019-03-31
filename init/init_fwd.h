#pragma once
#include <memory>
#include <stdlib.h>
#include <string>

class NetworkInitializer {
public:
  virtual void InitRegularWeights(double *w, size_t num, size_t layer,
                                  size_t src_size, size_t dst_size) = 0;
  virtual void InitConstantsWeights(double *w, size_t num, size_t layer,
                                    size_t src_size, size_t dst_size) = 0;
  virtual void InitConstants(double *c, size_t num, size_t layer) = 0;

  virtual std::string Name() const = 0;
  virtual ~NetworkInitializer() = default;

  static std::unique_ptr<NetworkInitializer> New(const std::string &function);
  static std::unique_ptr<NetworkInitializer> New();
};
