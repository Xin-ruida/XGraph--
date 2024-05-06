#include "../../shared/subgraph.cuh"
#include "../../shared/partitioner.cuh"
#include "../../shared/subgraph_generator.cuh"
#include "../../shared/gpu_error_check.cuh"
#include "../../shared/gpu_kernels.cuh"
#include "../../shared/subway_utilities.hpp"
#include "pr_dis.h"

void pr_compute(GraphStructure& graph, GraphStates<float>& states, Subgraph& subgraph, SubgraphGenerator<float>& subgen, float acc, bool isInternalNode[])
{
	Timer timer;
	timer.Start();

	uint gItr = 0;

	bool finished;
	bool* d_finished;
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));

	Partitioner partitioner;

	while (subgraph.numActiveNodes > 0)
	{
		gItr++;

		partitioner.partition(subgraph, subgraph.numActiveNodes);
		// a super iteration
		for (int i = 0; i < partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();

			//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			//mixLabels<<<partitioner.partitionNodeSize[i]/512 + 1 , 512>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);

			uint itr = 0;
			do
			{
				itr++;
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				pr_async_distributed << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (partitioner.partitionNodeSize[i],
					partitioner.fromNode[i],
					partitioner.fromEdge[i],
					subgraph.d_activeNodes,
					subgraph.d_activeNodesPointer,
					subgraph.d_activeEdgeList,
					graph.d_outDegree,
					states.d_value,
					states.d_delta,
					d_finished,
					acc,
					states.d_label1);


				cudaDeviceSynchronize();
				gpuErrorcheck(cudaPeekAtLastError());

				gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			} while (!(finished));

			//cout << itr << ((itr > 1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i << endl;
		}

		subgen.generate(graph, states, subgraph, acc);
		//YCY TODO:此处是个临时版本，这样判断效率比较低，最终版本要改generate函数
		finished = true;
		for (uint i = 0; i < subgraph.numActiveNodes; ++i)
		{
			if (isInternalNode[subgraph.activeNodes[i]])
			{
				finished = false;
				break;
			}
		}
		if (finished) break;
	}

	float runtime = timer.Finish();
	cout << "[" << getpid() << "]: " << "Processing finished in " << runtime / 1000 << " (s).\n";

	gpuErrorcheck(cudaMemcpy(states.delta, states.d_delta, graph.num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
}

bool pr_async(GraphStructure& graph, GraphStates<float>& states, float graph_value[], bool isInternalNode[], bool hasInit)
{
	float acc = 0.01;
	if (!hasInit)
	{
		float initPR = 0.15;
		for (unsigned int i = 0; i < graph.num_nodes; i++)
		{
			if (isInternalNode[i]) states.delta[i] = initPR;
			else states.delta[i] = 0;
			states.value[i] = graph_value[i];
		}

		gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
		gpuErrorcheck(cudaMemcpy(states.d_value, states.value, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
		gpuErrorcheck(cudaMemcpy(states.d_delta, states.delta, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
		gpuErrorcheck(cudaMemcpy(states.d_label1, isInternalNode, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	}
	else
	{
		bool end = true;
		for (uint i = 0; i < graph.num_nodes; i++)
		{
			if (!isInternalNode[i]) states.delta[i] = 0;
			else if (graph_value[i] > acc)
			{
				states.delta[i] = graph_value[i];
				end = false;
			}
		}
		if (end)
		{
			gpuErrorcheck(cudaMemcpy(states.value, states.d_value, graph.num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
			return true;
		}

		gpuErrorcheck(cudaMemcpy(states.d_delta, states.delta, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	}

	Subgraph subgraph(graph.num_nodes, graph.num_edges);
	SubgraphGenerator<float> subgen(graph);
	subgen.generate(graph, states, subgraph, acc);

	pr_compute(graph, states, subgraph, subgen, acc, isInternalNode);

	subgraph.FreeSubgraph();
	subgen.FreeSubgraphGenerator();

	return false;
}