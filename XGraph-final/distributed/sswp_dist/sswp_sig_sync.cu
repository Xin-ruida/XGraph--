#include "../../shared/timer.hpp"
#include "../../shared/subgraph.cuh"
#include "../../shared/partitioner.cuh"
#include "../../shared/subgraph_generator.cuh"
#include "../../shared/gpu_error_check.cuh"
#include "../../shared/gpu_kernels.cuh"
#include "../../shared/subway_utilities.hpp"
#include "sswp_sig.h"

void sswp_sig_sync(ArgumentParser arguments)
{
	cudaFree(0);

	Timer timer;
	timer.Start();

	GraphStructure graph;
	graph.ReadGraph(arguments.input);

	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime / 1000 << " (s).\n";

	GraphStates<uint> states(graph.num_nodes, true, false, graph.num_edges);
	states.ReadEdgeWeight(arguments.input + "w", graph.num_edges);

	for (unsigned int i = 0; i < graph.num_nodes; i++)
	{
		states.value[i] = 0;
		states.label1[i] = false;
		states.label2[i] = false;
	}
	states.value[arguments.sourceNode] = DIST_INFINITY;
	states.label1[arguments.sourceNode] = false;
	states.label2[arguments.sourceNode] = true;


	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_value, states.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	Subgraph subgraph(graph.num_nodes, graph.num_edges, true);

	SubgraphGenerator<uint> subgen(graph);

	subgen.generate(graph, states, subgraph);


	Partitioner partitioner;

	timer.Start();

	uint itr = 0;

	while (subgraph.numActiveNodes > 0)
	{
		itr++;

		partitioner.partition(subgraph, subgraph.numActiveNodes);
		// a super iteration
		for (int i = 0; i < partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeWeightList, subgraph.activeWeightList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(uint), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();

			moveUpLabels << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (subgraph.d_activeNodes, states.d_label1, states.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);

			sswp_kernel << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (partitioner.partitionNodeSize[i],
				partitioner.fromNode[i],
				partitioner.fromEdge[i],
				subgraph.d_activeNodes,
				subgraph.d_activeNodesPointer,
				subgraph.d_activeEdgeList,
				subgraph.d_activeWeightList,
				graph.d_outDegree,
				states.d_value,
				//d_finished,
				states.d_label1,
				states.d_label2);

			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());
		}

		subgen.generate(graph, states, subgraph);

	}

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime / 1000 << " (s).\n";

	cout << "Number of iterations = " << itr << endl;

	gpuErrorcheck(cudaMemcpy(states.value, states.d_value, graph.num_nodes * sizeof(uint), cudaMemcpyDeviceToHost));

	utilities::PrintResults(states.value, min(30, graph.num_nodes));

	if (arguments.hasOutput)
		utilities::SaveResults(arguments.output, states.value, graph.num_nodes);
}

