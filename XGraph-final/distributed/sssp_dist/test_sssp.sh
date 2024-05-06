#!/bin/bash

mpirun --allow-run-as-root -n 2 -hostfile hostfile ./sssp --input ../../../data/LiveJournal.bcsr --device 0 --source 1 --dist 1