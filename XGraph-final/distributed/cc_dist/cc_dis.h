#include "../../shared/globals.hpp"
#include "../../shared/graph.cuh"
#include "../../shared/argument_parsing.cuh"
#include "../../shared/timer.hpp"
#include "unistd.h"

extern "C" bool cc_async(GraphStructure& graph, GraphStates<uint>& states, uint graph_value[], bool hasInit = true);
extern "C" void part(GraphStructure graph, GraphStructure graph_cut[], GraphStates<uint> states, GraphStates<uint> states_cut[], uint n);
extern void cc_dis_async(ArgumentParser arguments);