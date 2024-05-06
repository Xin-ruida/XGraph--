#!/bin/sh

mpirun --allow-run-as-root -n 2 -hostfile hostfile ./bfs --input ../../../data/LiveJournal.bcsr --device 0 --source 1 --dist 1
