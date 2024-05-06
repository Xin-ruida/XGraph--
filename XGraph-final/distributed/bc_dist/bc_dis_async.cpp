#include <iostream>
#include <mpi.h>

using std::cout;
using std::cerr;
using std::endl;

#include "bc_dis.h"
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
        cerr << "MPI error calling \""#call"\"\n"; \
        cout << "Test FAILED\n"; \
        MPI_Abort(MPI_COMM_WORLD, -1); }

void bc_dis_async(ArgumentParser arguments)
{
    MPI_Init(NULL, NULL);
    int world_size,world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    Timer timer;
    timer.Start();
    // 读原图
    GraphStructure AllGraph;
    AllGraph.ReadGraph(arguments.input);
    GraphStates<uint> AllStates(AllGraph.num_nodes, true, true, true);

    float readtime = timer.Finish();
    cout << "[" << getpid() << "]: " << "Graph Reading finished in " << readtime / 1000 << " (s).\n";

    uint n = (uint)world_size;
    GraphStructure graph_cut[n];
    GraphStates<uint> states_cut[n];
    // 图划分
    bc_part(AllGraph, graph_cut, AllStates, states_cut, n);
    AllGraph.FreeGraph();
    AllStates.FreeStates();

    // 值初始化
    uint *graph_value = new uint[AllGraph.num_nodes];
    for(uint i=0;i< AllGraph.num_nodes;i++)
    {
        graph_value[i] = DIST_INFINITY;
    }

    GraphStructure graph;
    graph.ReadGraphFromGraph(graph_cut[world_rank]);
    GraphStates<uint> states(graph.num_nodes, true, true, true);

    cout << "[" << getpid() << "]: " << "bfs start: " << endl;
    bc_bfs_async(graph, states, graph_value, false, arguments.sourceNode);

    int iter = 1;
    int myFlag;
    int allFlags[n];
    while (true)
    {
        MPI_Allreduce(states.value, graph_value, AllGraph.num_nodes, MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
        cout << "[" << getpid() << "]: " << "communicate " << iter++ << " times" << endl;

        bool end = bc_bfs_async(graph, states, graph_value);
        if (end) myFlag = 0;
        else myFlag = 1;

        MPI_Allgather(&myFlag, 1, MPI_INT, allFlags, 1, MPI_INT, MPI_COMM_WORLD);

        end = true;
        for (int i = 0; i < n; ++i)
        {
            if (allFlags[i] == 1)
            {
                end = false;
                break;
            }
        }
        if (end) break;
    }

    int level = find_max_distance(graph, states);
    cout << "[" << getpid() << "]: " << level << endl;

    cout << "[" << getpid() << "]: " << "compute sigma start: " << endl;
    uint *graph_sigma = new uint[AllGraph.num_nodes];
    for(uint i=0;i< AllGraph.num_nodes;i++)
    {
        graph_sigma[i] = 0;
    }
    graph_sigma[arguments.sourceNode] = 1;
    int start = 0;
	while(start < level) {
		bc_sigma_async(graph, states, graph_sigma, start);
        MPI_Allreduce(states.sigma, graph_value, AllGraph.num_nodes, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        for(uint i=0;i< AllGraph.num_nodes;i++)
        {
           graph_sigma[i] = graph_value[i] - (world_size - 1) * graph_sigma[i];
        }
		start++;
	}
    start = -1;
    bc_sigma_async(graph, states, graph_sigma, start);

    cout << "[" << getpid() << "]: " << "compute bc start: " << endl;
    float* graph_delta = (float*)graph_sigma;
    for(uint i=0;i< AllGraph.num_nodes;i++)
    {
        graph_delta[i] = 0;
    }
    while(level >= 1) {
		level--;
		bc_async(graph, states, graph_delta, level);
        MPI_Allreduce(states.delta, graph_delta, AllGraph.num_nodes, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
	}

    if (world_rank == 0) {
        for (uint i = 0; i < graph.num_nodes && i < 30; i++)
        {
            cout << i << ":" << states.value[i] << " " << states.sigma[i] << " " << graph_delta[i] << ", ";
        }
        cout << endl;
    }

    MPI_CHECK(MPI_Finalize());
}
