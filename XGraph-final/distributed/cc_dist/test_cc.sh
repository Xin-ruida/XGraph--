#!/bin/sh

mpirun --allow-run-as-root -n 2 -hostfile hostfile ./cc --input ../../../data/soc-pokec-relationships.bcsr --device 0 --dist 1