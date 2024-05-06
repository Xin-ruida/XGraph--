
CC=/usr/bin/mpicxx
NC=/usr/local/cuda/bin/nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-gencode arch=compute_60,code=sm_60
# NFLAGS=-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75

ROOT_DIR= $(shell pwd)
CXX?= g++
CXXFLAGS?= -O3 -Wall -std=c++11 -g -fopenmp -I$(ROOT_DIR)

#compilers
ifdef CILK
PCC = g++
PCFLAGS = -std=c++14 -fcilkplus -lcilkrts -O3 -DCILK $(INTT) $(INTE) $(CODE) $(PD) $(MEM)
PLFLAGS = -fcilkplus -lcilkrts

else ifdef MKLROOT
PCC = icpc
PCFLAGS = -std=c++14 -O3 -DCILKP $(INTT) $(INTE) $(CODE) $(PD) $(MEM)

else ifdef OPENMP
PCC = g++
PCFLAGS = -std=c++14 -fopenmp -march=native -O3 -DOPENMP $(INTT) $(INTE) $(CODE) $(PD) $(MEM)

else
PCC = g++
PCFLAGS = -std=c++14 -O3 $(INTT) $(INTE) $(CODE) $(PD) $(MEM)
endif

SHARED=../../shared
INCLUDES = ../../cpu
DISK = ../../diskcore

DEP=$(SHARED)/argument_parsing.o $(SHARED)/graph.o $(SHARED)/timer.o $(SHARED)/gpu_kernels.o $(SHARED)/partitioner.o $(SHARED)/subway_utilities.o $(SHARED)/subgraph_generator.o $(SHARED)/subgraph.o
HEADERS= $(DISK)/atomic.hpp $(DISK)/bigvector.hpp $(DISK)/bitmap.hpp $(DISK)/constants.hpp $(DISK)/filesystem.hpp $(DISK)/graph.hpp $(DISK)/partition.hpp $(DISK)/queue.hpp $(DISK)/time.hpp $(DISK)/type.hpp

all: part sig dis pr PageRank pagerank

part: part.cu 
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o part.o -c part.cu $(CFLAGS)

sig: pr_sig_sync.cu pr_sig_async.cu
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o pr_sig_sync.o -c pr_sig_sync.cu $(CFLAGS)
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o pr_sig_async.o -c pr_sig_async.cu $(CFLAGS)

dis: pr-async.cu pr_dis_async.cpp
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o pr-async.o -c pr-async.cu $(CFLAGS)
	$(CC) -I../../common/inc -o pr_dis_async.o -c pr_dis_async.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart

pr: pr.cpp part.o pr_sig_sync.o pr_sig_async.o pr-async.o pr_dis_async.o
	$(CC) -I../../common/inc -o pr.o -c pr.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
	$(CC) -o pr $(DEP) part.o pr_sig_sync.o pr_sig_async.o pr-async.o pr_dis_async.o pr.o -L/usr/local/cuda/lib64 -lcudart -lpthread

PageRank : PageRank.C
	$(PCC) $(PCFLAGS) -I $(INCLUDES) -o $@ $<

pagerank : pagerank.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

clean:
	rm part.o
	rm pr_sig_sync.o pr_sig_async.o
	rm pr-async.o pr_dis_async.o 
	rm pr.o pr
	rm PageRank pagerank
