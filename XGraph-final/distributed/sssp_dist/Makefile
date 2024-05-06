
CC=/usr/bin/mpicxx
NC=/usr/local/cuda/bin/nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-gencode arch=compute_60,code=sm_60
# NFLAGS=-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75

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

DEP=$(SHARED)/argument_parsing.o $(SHARED)/graph.o $(SHARED)/timer.o $(SHARED)/gpu_kernels.o $(SHARED)/partitioner.o $(SHARED)/subway_utilities.o $(SHARED)/subgraph_generator.o $(SHARED)/subgraph.o


all: part sig dis sssp BellmanFord

part: part.cu
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o part.o -c part.cu $(CFLAGS)

sig: sssp_sig_sync.cu sssp_sig_async.cu
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o sssp_sig_sync.o -c sssp_sig_sync.cu $(CFLAGS)
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o sssp_sig_async.o -c sssp_sig_async.cu $(CFLAGS)

dis: sssp-async.cu sssp_dis_async.cpp
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o sssp-async.o -c sssp-async.cu $(CFLAGS)
	$(CC) -I../../common/inc -o sssp_dis_async.o -c sssp_dis_async.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart

sssp: sssp.cpp part.o sssp_sig_sync.o sssp_sig_async.o sssp-async.o sssp_dis_async.o
	$(CC) -I../../common/inc -o sssp.o -c sssp.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
	$(CC) -o sssp $(DEP) part.o sssp_sig_sync.o sssp_sig_async.o sssp-async.o sssp_dis_async.o sssp.o -L/usr/local/cuda/lib64 -lcudart -lpthread

#sssp_dis_sync: sssp_dis_sync.cpp sssp-sync.cu part.o
#	$(CC) -I../../common/inc -o sssp_dis_sync.o -c sssp_dis_sync.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o sssp-sync.o -c sssp-sync.cu $(CFLAGS)
#	$(CC) -o sssp_dis_sync $(DEP) sssp-sync.o sssp_dis_sync.o part.o -L/usr/local/cuda/lib64 -lcudart -lpthread

#sssp_dis_async: sssp_dis_async.cpp sssp-async.cu part.o 
#	$(CC) -I../../common/inc -o sssp_dis_async.o -c sssp_dis_async.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o sssp-async.o -c sssp-async.cu $(CFLAGS)
#	$(CC) -o sssp_dis_async $(DEP) sssp-async.o sssp_dis_async.o part.o -L/usr/local/cuda/lib64 -lcudart -lpthread

BellmanFord : BellmanFord.C
	$(PCC) $(PCFLAGS) -I $(INCLUDES) -o $@ $<

clean:
	rm part.o
	rm sssp_sig_sync.o sssp_sig_async.o
	rm sssp-async.o sssp_dis_async.o
	rm sssp.o sssp
	rm BellmanFord
