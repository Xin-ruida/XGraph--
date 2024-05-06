#include "../../shared/globals.hpp"
#include "../../shared/graph.cuh"
#include "../../shared/argument_parsing.cuh"
#include "../../shared/timer.hpp"
#include "unistd.h"

extern "C" bool bc_bfs_async(GraphStructure & graph, GraphStates<uint>&states, uint graph_value[], bool hasInit = true, int sourceNode = 0);
extern "C" int find_max_distance(GraphStructure& graph, GraphStates<uint>& states);
extern "C" void bc_sigma_async(GraphStructure & graph, GraphStates<uint>&states, uint graph_sigma[], int level);
extern "C" void bc_async(GraphStructure & graph, GraphStates<uint>&states, float graph_delta[], int level);
extern "C" void bc_part(GraphStructure graph, GraphStructure graph_cut[], GraphStates<uint> states, GraphStates<uint> states_cut[], uint n);
extern void bc_dis_async(ArgumentParser arguments);
