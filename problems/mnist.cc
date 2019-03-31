#include "problems/mnist.h"
#include <arpa/inet.h>
#include <assert.h>

uint32_t ReadU32(FILE *in) {
  uint32_t ret;
  fread(&ret, 1, sizeof(ret), in);
  return ntohl(ret);
}

std::unique_ptr<MnistData> MnistData::Load(bool train) {
  std::string prefix = train ? "data/train" : "data/t10k";
  FILE *data = fopen((prefix + "-images-idx3-ubyte").c_str(), "r");
  FILE *labels = fopen((prefix + "-labels-idx1-ubyte").c_str(), "r");
  assert(data);
  assert(labels);
  uint32_t magic = ReadU32(data);
  assert(magic == 0x803);
  magic = ReadU32(labels);
  assert(magic == 0x801);

  auto ret = std::make_unique<MnistData>();
  size_t sized = ReadU32(data);
  size_t sizel = ReadU32(labels);
  assert(sized == sizel);

  ret->vals.resize(sizel);
  int num = fread(ret->vals.data(), 1, sizel, labels);
  assert(num == int(sized));

  size_t t = ReadU32(data);
  size_t u = ReadU32(data);
  assert(t == 28);
  assert(u == 28);

  ret->data.resize(sized * ret->NumInputs());
  num = fread(ret->data.data(), 1, sized * ret->NumInputs(), data);
  assert(num == int(sized * ret->NumInputs()));

  fclose(data);
  fclose(labels);
  return ret;
}
