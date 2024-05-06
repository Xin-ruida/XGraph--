#!/bin/bash

mpirun --allow-run-as-root -n 2 -hostfile hostfile ./pr --input ../../../data/LiveJournal.bcsr --device 0 --dist 1
./pagerank ../../../data/LiveJournal.pl 30 14