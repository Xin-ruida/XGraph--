
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


all: part sig dis bfs BFS

part: bfs_part.cu 
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bfs_part.o -c bfs_part.cu $(CFLAGS)

sig: bfs_sig_sync.cu bfs_sig_async.cu
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bfs_sig_sync.o -c bfs_sig_sync.cu $(CFLAGS)
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bfs_sig_async.o -c bfs_sig_async.cu $(CFLAGS)

dis: bfs-async.cu bfs_dis_async.cpp
	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bfs-async.o -c bfs-async.cu $(CFLAGS)
	$(CC) -I../../common/inc -o bfs_dis_async.o -c bfs_dis_async.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart

#sync: bfs-sync.cu bfs_sig_sync.cu bfs_dis_sync.cpp
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bfs-sync.o -c bfs-sync.cu $(CFLAGS)
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bfs_sig_sync.o -c bfs_sig_sync.cu $(CFLAGS)
#	$(CC) -I../../common/inc -o bfs_dis_sync.o -c bfs_dis_sync.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart

#async: bfs-async.cu bfs_sig_async.cu bfs_dis_async.cpp
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bfs-async.o -c bfs-async.cu $(CFLAGS)
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bfs_sig_async.o -c bfs_sig_async.cu $(CFLAGS)
#	$(CC) -I../../common/inc -o bfs_dis_async.o -c bfs_dis_async.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart

bfs: bfs.cpp bfs_part.o bfs_sig_sync.o bfs_sig_async.o bfs-async.o bfs_dis_async.o
	$(CC) -I../../common/inc -o bfs.o -c bfs.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
	$(CC) -o bfs $(DEP) bfs_part.o bfs_sig_sync.o bfs_sig_async.o bfs-async.o bfs_dis_async.o bfs.o -L/usr/local/cuda/lib64 -lcudart -lpthread
	
#bfs_dis_sync: bfs_dis_sync.cpp bfs-sync.cu bfs_part.o
#	$(CC) -I../../common/inc -o bfs_dis_sync.o -c bfs_dis_sync.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bfs-sync.o -c bfs-sync.cu $(CFLAGS)
#	$(CC) -o bfs_dis_sync $(DEP) bfs-sync.o bfs_dis_sync.o bfs_part.o -L/usr/local/cuda/lib64 -lcudart -lpthread

#bfs_dis_async: bfs_dis_async.cpp bfs-async.cu bfs_part.o 
#	$(CC) -I../../common/inc -o bfs_dis_async.o -c bfs_dis_async.cpp $(CFLAGS) -L/usr/local/cuda/lib64 -lcudart
#	$(NC) -ccbin g++ -I../../common/inc -m64  $(NFLAGS) -o bfs-async.o -c bfs-async.cu $(CFLAGS)
#	$(CC) -o bfs_dis_async $(DEP) bfs-async.o bfs_dis_async.o bfs_part.o -L/usr/local/cuda/lib64 -lcudart -lpthread

BFS : BFS.C
	$(PCC) $(PCFLAGS) -I $(INCLUDES) -o $@ $<

clean:
	rm bfs_part.o
	rm bfs_sig_sync.o bfs_sig_async.o
	rm bfs-async.o bfs_dis_async.o 
	rm bfs.o bfs
	rm BFS
