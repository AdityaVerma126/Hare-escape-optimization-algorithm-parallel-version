CXX   ?= g++
NVCC  ?= nvcc
ARCH  ?= native            # e.g. ARCH=sm_86 if your nvcc is older than 11.5
CXXFLAGS = -O3 -std=c++17 -ffp-contract=off
.PHONY: all clean
all: heo_seq heo_omp heo_cuda
heo_seq: heo_seq.cpp heo_common.h heo_main.h
	$(CXX) $(CXXFLAGS) -o $@ $<
heo_omp: heo_omp.cpp heo_common.h heo_main.h
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $<
heo_cuda: heo_cuda.cu heo_common.h heo_main.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) --fmad=false -o $@ $<
clean:
	rm -f heo_seq heo_omp heo_cuda
