#include "../../shared/globals.hpp"
#include "../../shared/graph.cuh"
#include "../../shared/argument_parsing.cuh"
#include "../../shared/timer.hpp"
#include "unistd.h"

extern "C" bool sswp_async(GraphStructure & graph, GraphStates<uint>&states, uint graph_value[], bool hasInit = true, int sourceNode = 0);
extern "C" void part(GraphStructure graph, GraphStructure graph_cut[], GraphStates<uint> states, GraphStates<uint> states_cut[], uint n);
extern void sswp_dis_async(ArgumentParser arguments);