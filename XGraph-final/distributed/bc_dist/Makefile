
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


all: part sig dis bc BC

part: bc_part.cu 
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bc_part.o -c bc_part.cu $(CFLAGS)

sig: bc_sig_sync.cu bc_sig_async.cu
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bc_sig_sync.o -c bc_sig_sync.cu $(CFLAGS)
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bc_sig_async.o -c bc_sig_async.cu $(CFLAGS)

dis: bc-async.cu bc_dis_async.cpp
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bc-async.o -c bc-async.cu $(CFLAGS)
	$(CC) -I../../common/inc -o bc_dis_async.o -c bc_dis_async.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart

#sync: bc-sync.cu bc_sig_sync.cu bc_dis_sync.cpp
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bc-sync.o -c bc-sync.cu $(CFLAGS)
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bc_sig_sync.o -c bc_sig_sync.cu $(CFLAGS)
#	$(CC) -I../../common/inc -o bc_dis_sync.o -c bc_dis_sync.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart

#async: bc-async.cu bc_sig_async.cu bc_dis_async.cpp
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bc-async.o -c bc-async.cu $(CFLAGS)
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bc_sig_async.o -c bc_sig_async.cu $(CFLAGS)
#	$(CC) -I../../common/inc -o bc_dis_async.o -c bc_dis_async.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart

bc: bc.cpp bc_part.o bc_sig_sync.o bc_sig_async.o bc-async.o bc_dis_async.o
	$(CC) -I../../common/inc -o bc.o -c bc.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
	$(CC) -o bc $(DEP) bc_part.o bc_sig_sync.o bc_sig_async.o bc-async.o bc_dis_async.o bc.o -L/usr/local/cuda/lib64 -lcudart -lpthread
	
#bc_dis_sync: bc_dis_sync.cpp bc-sync.cu bc_part.o
#	$(CC) -I../../common/inc -o bc_dis_sync.o -c bc_dis_sync.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bc-sync.o -c bc-sync.cu $(CFLAGS)
#	$(CC) -o bc_dis_sync $(DEP) bc-sync.o bc_dis_sync.o bc_part.o -L/usr/local/cuda/lib64 -lcudart -lpthread

#bc_dis_async: bc_dis_async.cpp bc-async.cu bc_part.o 
#	$(CC) -I../../common/inc -o bc_dis_async.o -c bc_dis_async.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bc-async.o -c bc-async.cu $(CFLAGS)
#	$(CC) -o bc_dis_async $(DEP) bc-async.o bc_dis_async.o bc_part.o -L/usr/local/cuda/lib64 -lcudart -lpthread

BC : BC.C
	$(PCC) $(PCFLAGS) -I $(INCLUDES) -o $@ $<

clean:
	rm bc_part.o
	rm bc_sig_sync.o bc_sig_async.o
	rm bc-async.o bc_dis_async.o 
	rm bc.o bc
	rm BC
