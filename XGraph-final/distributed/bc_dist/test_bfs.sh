#!/bin/sh

mpirun -n 2 -hostfile hostfile ./bc_dis_async --input /hzy/hzy/dataset/uk-2007.bcsr
