#include "../../shared/subgraph.cuh"
#include "../../shared/partitioner.cuh"
#include "../../shared/subgraph_generator.cuh"
#include "../../shared/gpu_error_check.cuh"
#include "../../shared/gpu_kernels.cuh"
#include "../../shared/subway_utilities.hpp"
#include "sssp_dis.h"

void sssp_compute(GraphStructure& graph, GraphStates<uint>& states, Subgraph& subgraph, SubgraphGenerator<uint>& subgen)
{
	Timer timer;
	timer.Start();

	uint gItr = 0;

	bool finished;
	bool* d_finished;
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));

	Partitioner partitioner;

	// 子图迭代
	while (subgraph.numActiveNodes > 0)
	{
		gItr++;

		partitioner.partition(subgraph, subgraph.numActiveNodes);
		// a super iteration
		for (int i = 0; i < partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeWeightList, subgraph.activeWeightList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(uint), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();

			//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels << <partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (subgraph.d_activeNodes, states.d_label1, states.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);

			uint itr = 0;
			do
			{
				itr++;
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				sssp_async << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (partitioner.partitionNodeSize[i],
					partitioner.fromNode[i],
					partitioner.fromEdge[i],
					subgraph.d_activeNodes,
					subgraph.d_activeNodesPointer,
					subgraph.d_activeEdgeList,
					subgraph.d_activeWeightList,
					graph.d_outDegree,
					states.d_value,
					d_finished,
					(itr % 2 == 1) ? states.d_label1 : states.d_label2,
					(itr % 2 == 1) ? states.d_label2 : states.d_label1);

				cudaDeviceSynchronize();
				gpuErrorcheck(cudaPeekAtLastError());

				gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			} while (!(finished));

			//cout << itr << ((itr > 1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i << endl;
		}

		subgen.generate(graph, states, subgraph);

	}

	float runtime = timer.Finish();
	cout << "[" << getpid() << "]: " << "Processing finished in " << runtime / 1000 << " (s).\n";

	gpuErrorcheck(cudaMemcpy(states.value, states.d_value, graph.num_nodes * sizeof(uint), cudaMemcpyDeviceToHost));
}

bool sssp_async(GraphStructure& graph, GraphStates<uint>& states, uint graph_value[], bool hasInit, int sourceNode)
{
	if (!hasInit)
	{
		for (unsigned int i = 0; i < graph.num_nodes; i++)
		{
			states.value[i] = graph_value[i];
		}
		states.value[sourceNode] = 0;

		gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
		gpuErrorcheck(cudaMemcpy(states.d_value, states.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	}
	for (unsigned int i = 0; i < graph.num_nodes; i++)
	{
		states.label1[i] = true;
		states.label2[i] = false;
	}
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	// 单机子图划分
	Subgraph subgraph(graph.num_nodes, graph.num_edges, true);
	SubgraphGenerator<uint> subgen(graph);
	subgen.generate(graph, states, subgraph);

	for (unsigned int i = 0; i < graph.num_nodes; i++)
	{
		states.label1[i] = false;
	}
	if (!hasInit) states.label1[sourceNode] = true;
	else
	{
		bool end = true;
		for (uint i = 0; i < graph.num_nodes; i++)
		{
			if (graph_value[i] < states.value[i])
			{
				states.value[i] = graph_value[i];
				states.label1[i] = true;
				end = false;
			}
		}
		if (end)
		{
			subgraph.FreeSubgraph();
			subgen.FreeSubgraphGenerator();
			return true;
		}

		gpuErrorcheck(cudaMemcpy(states.d_value, states.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	}

	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	sssp_compute(graph, states, subgraph, subgen);

	subgraph.FreeSubgraph();
	subgen.FreeSubgraphGenerator();

	return false;
}

