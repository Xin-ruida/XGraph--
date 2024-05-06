#include "../../shared/globals.hpp"
#include "../../shared/graph.cuh"
#include "../../shared/argument_parsing.cuh"
#include "../../shared/timer.hpp"
#include "unistd.h"

extern "C" bool pr_async(GraphStructure& graph, GraphStates<float>& states, float graph_value[], bool isInternalNode[], bool hasInit = true);
extern "C" void part(GraphStructure graph, GraphStructure graph_cut[], GraphStates<float> states, GraphStates<float> states_cut[], uint n, bool isInternalNode[], int workerID);
extern void pr_dis_async(ArgumentParser arguments);
