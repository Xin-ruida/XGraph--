
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


all: part sig dis cc Components

part: part.cu
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o part.o -c part.cu $(CFLAGS)

sig: cc_sig_sync.cu cc_sig_async.cu
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o cc_sig_sync.o -c cc_sig_sync.cu $(CFLAGS)
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o cc_sig_async.o -c cc_sig_async.cu $(CFLAGS)

dis: cc-async.cu cc_dis_async.cpp
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o cc-async.o -c cc-async.cu $(CFLAGS)
	$(CC) -I../../common/inc -o cc_dis_async.o -c cc_dis_async.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart

cc: cc.cpp part.o cc_sig_sync.o cc_sig_async.o cc-async.o cc_dis_async.o
	$(CC) -I../../common/inc -o cc.o -c cc.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
	$(CC) -o cc $(DEP) part.o cc_sig_sync.o cc_sig_async.o cc-async.o cc_dis_async.o cc.o -L/usr/local/cuda/lib64 -lcudart -lpthread

#cc_dis_sync: cc_dis_sync.cpp cc-sync.cu part.o
#	$(CC) -I../../common/inc -o cc_dis_sync.o -c cc_dis_sync.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o cc-sync.o -c cc-sync.cu $(CFLAGS)
#	$(CC) -o cc_dis_sync $(DEP) cc-sync.o cc_dis_sync.o part.o -L/usr/local/cuda/lib64 -lcudart -lpthread

#cc_dis_async: cc_dis_async.cpp cc-async.cu part.o 
#	$(CC) -I../../common/inc -o cc_dis_async.o -c cc_dis_async.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o cc-async.o -c cc-async.cu $(CFLAGS)
#	$(CC) -o cc_dis_async $(DEP) cc-async.o cc_dis_async.o part.o -L/usr/local/cuda/lib64 -lcudart -lpthread

Components : Components.C
	$(PCC) $(PCFLAGS) -I $(INCLUDES) -o $@ $<

clean:
	rm part.o
	rm cc_sig_sync.o cc_sig_async.o
	rm cc-async.o cc_dis_async.o
	rm cc.o cc
	rm Components
