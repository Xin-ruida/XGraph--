#include <iostream>
#include <mpi.h>

using std::cout;
using std::cerr;
using std::endl;

#include "pr_dis.h"
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
        cerr << "MPI error calling \""#call"\"\n"; \
        cout << "Test FAILED\n"; \
        MPI_Abort(MPI_COMM_WORLD, -1); }

void pr_dis_async(ArgumentParser arguments)
{
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    Timer timer;
    timer.Start();
    // ��ԭͼ
    GraphStructure AllGraph;
    AllGraph.ReadGraph(arguments.input);
    //GraphStates<float> AllStates(AllGraph.num_nodes, false, true);
    GraphStates<float> AllStates(AllGraph.num_nodes, true, true);

    float readtime = timer.Finish();
    cout << "[" << getpid() << "]: " << "Graph Reading finished in " << readtime / 1000 << " (s).\n";

    uint n = (uint)world_size;
    GraphStructure graph_cut[n];
    GraphStates<float> states_cut[n];
    // ͼ����
    bool* isInternalNode = new bool[AllGraph.num_nodes];
    for (uint i = 0; i < AllGraph.num_nodes; i++)
        isInternalNode[i] = false;
    part(AllGraph, graph_cut, AllStates, states_cut, n, isInternalNode, world_rank);
    AllGraph.FreeGraph();
    AllStates.FreeStates();

    // ֵ��ʼ��
    float* graph_value = new float[AllGraph.num_nodes];
    for (uint i = 0; i < AllGraph.num_nodes; i++)
    {
        graph_value[i] = 0;
    }

    GraphStructure graph;
    graph.ReadGraphFromGraph(graph_cut[world_rank]);
    //GraphStates<float> states(graph.num_nodes, false, true);
    GraphStates<float> states(graph.num_nodes, true, true);
    
    pr_async(graph, states, graph_value, isInternalNode, false);

    int iter = 1;
    int myFlag;
    int allFlags[n];
    while (true)
    {
        MPI_Allreduce(states.delta, graph_value, AllGraph.num_nodes, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        cout << "[" << getpid() << "]: " << "communicate " << iter++ << " times" << endl;

        bool end = pr_async(graph, states, graph_value, isInternalNode);
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

    MPI_Allreduce(states.value, graph_value, AllGraph.num_nodes, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    if (world_rank == 0) {
        for (uint i = 0; i < graph.num_nodes && i < 30; i++)
        {
            cout << i << ":" << graph_value[i] << " ";
        }
        cout << endl;
    }

    MPI_CHECK(MPI_Finalize());
}
