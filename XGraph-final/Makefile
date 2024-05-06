CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60

SHARED=shared
APPS=apps
TOOLS=tools
BFS=distributed/bfs_dist
CCDIST=distributed/cc_dist
SSSP=distributed/sssp_dist
SSWP=distributed/sswp_dist
PR=distributed/pr_dist
BCDIST=distributed/bc_dist

DEP=$(SHARED)/timer.o $(SHARED)/argument_parsing.o $(SHARED)/graph.o $(SHARED)/subgraph.o $(SHARED)/partitioner.o $(SHARED)/subgraph_generator.o $(SHARED)/gpu_kernels.o $(SHARED)/subway_utilities.o $(SHARED)/test.o  

# all: make1 make2 make3 bfs-sync cc-sync sssp-sync sswp-sync pr-sync bfs-async cc-async sssp-async sswp-async pr-async

all: make1 make2 make3 bfs-w cc-w bc-w pr-w sssp-w bfs cc sssp sswp pr bc

make1:
	make -C $(SHARED)

make2:
	make -C $(APPS)

make3:
	make -C $(TOOLS)

bfs-w: $(APPS)/bfs-w.o $(DEP)
	$(NC) $(APPS)/bfs-w.o $(DEP) -o bfs-w $(CFLAGS) $(NFLAGS)

bc-w: $(APPS)/bc-w.o $(DEP)
	$(NC) $(APPS)/bc-w.o $(DEP) -o bc-w $(CFLAGS) $(NFLAGS)

cc-w: $(APPS)/cc-w.o $(DEP)
	$(NC) $(APPS)/cc-w.o $(DEP) -o cc-w $(CFLAGS) $(NFLAGS)

pr-w: $(APPS)/pr-w.o $(DEP)
	$(NC) $(APPS)/pr-w.o $(DEP) -o pr-w $(CFLAGS) $(NFLAGS)
	
sssp-w: $(APPS)/sssp-w.o $(DEP)
	$(NC) $(APPS)/sssp-w.o $(DEP) -o sssp-w $(CFLAGS) $(NFLAGS)

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

bc:
	make -C $(BCDIST)
	
clean:
	make -C $(SHARED) clean
	make -C $(APPS) clean
	make -C $(TOOLS) clean
	rm -f bfs-w cc-w bc-w pr-w sssp-w
	make -C $(BFS) clean
	make -C $(CCDIST) clean
	make -C $(SSSP) clean
	make -C $(SSWP) clean
	make -C $(PR) clean
	make -C $(BCDIST) clean
