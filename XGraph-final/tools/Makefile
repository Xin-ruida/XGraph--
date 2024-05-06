
CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60

ROOT_DIR= $(shell pwd)
CXX?= g++
CXXFLAGS?= -O3 -Wall -std=c++11 -g -fopenmp -I$(ROOT_DIR)

SHARED=../shared
DISK = ../diskcore

HEADERS= $(DISK)/atomic.hpp $(DISK)/bigvector.hpp $(DISK)/bitmap.hpp $(DISK)/constants.hpp $(DISK)/filesystem.hpp $(DISK)/graph.hpp $(DISK)/partition.hpp $(DISK)/queue.hpp $(DISK)/time.hpp $(DISK)/type.hpp

all: converter preprocess

converter: converter.cpp
	$(CC) converter.cpp -o converter $(CFLAGS)

preprocess: preprocess.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

clean:
	rm -f converter preprocess
