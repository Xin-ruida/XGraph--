CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60

SHARED=../shared
BFS=bfs_dist
CCDIST=cc_dist
SSSP=sssp_dist
SSWP=sswp_dist
PR=pr_dist

TOOLS=../tools

DEP=$(SHARED)/timer.o $(SHARED)/argument_parsing.o $(SHARED)/graph.o $(SHARED)/subgraph.o $(SHARED)/partitioner.o $(SHARED)/subgraph_generator.o $(SHARED)/gpu_kernels.o $(SHARED)/utilities.o $(SHARED)/test.o

all: make1 make2 bfs cc sssp sswp pr

make1:
	make -C $(SHARED)

make2: 
	make -C $(TOOLS)

bfs:
	make -C $(BFS)

cc:
	make -C $(CCDIST)

sssp:
	make -C $(SSSP)

sswp:
	make -C $(SSWP)

pr:
	make -C $(PR)

clean:
	make -C $(BFS) clean
	make -C $(CCDIST) clean
	make -C $(SSSP) clean
	make -C $(SSWP) clean
	make -C $(PR) clean
	make -C $(SHARED) clean
	make -C $(TOOLS) clean
