CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60

ALL = bfs-w.o cc-w.o pr-w.o bc-w.o sssp-w.o
all: $(ALL)

bfs-w.o: bfs-w.cu
	$(NC) -c bfs-w.cu $(CFLAGS) $(NFLAGS)

bc-w.o: bc-w.cu
	$(NC) -c bc-w.cu $(CFLAGS) $(NFLAGS)

cc-w.o: cc-w.cu
	$(NC) -c cc-w.cu $(CFLAGS) $(NFLAGS)	

sssp-w.o: sssp-w.cu
	$(NC) -c sssp-w.cu $(CFLAGS) $(NFLAGS)

pr-w.o: pr-w.cu
	$(NC) -c pr-w.cu $(CFLAGS) $(NFLAGS)	

.PHONY : clean		

clean:
	rm $(ALL)


