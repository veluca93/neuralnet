CXXFLAGS:=-O3 -Wall -std=c++17 -flto -g -march=native -I. -fno-exceptions \
	$(pkg-config --cflags gflags) -fopenmp
LDFLAGS:=-flto -lpthread $(shell pkg-config --libs gflags)
CXX:=clang++
