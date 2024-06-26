#include "../../shared/timer.hpp"
#include "../../shared/subgraph.cuh"
#include "../../shared/partitioner.cuh"
#include "../../shared/subgraph_generator.cuh"
#include "../../shared/gpu_error_check.cuh"
#include "../../shared/gpu_kernels.cuh"
#include "../../shared/subway_utilities.hpp"
#include "sswp_sig.h"

void sswp_sig_async(ArgumentParser arguments)
{
	cudaFree(0);

	Timer timer;
	timer.Start();

	GraphStructure graph;
	graph.ReadGraph(arguments.input);

	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime / 1000 << " (s).\n";

	//for(unsigned int i=0; i<100; i++)
	//	cout << graph.edgeList[i].end << " " << graph.edgeList[i].w8;

	GraphStates<uint> states(graph.num_nodes, true, false, graph.num_edges);
	states.ReadEdgeWeight(arguments.input + "w", graph.num_edges);

	for (unsigned int i = 0; i < graph.num_nodes; i++)
	{
		states.value[i] = 0;
		states.label1[i] = true;
		states.label2[i] = false;
	}
	states.value[arguments.sourceNode] = DIST_INFINITY;
	//graph.label[arguments.sourceNode] = true;


	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_value, states.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	Subgraph subgraph(graph.num_nodes, graph.num_edges, true);

	SubgraphGenerator<uint> subgen(graph);

	subgen.generate(graph, states, subgraph);

	for (unsigned int i = 0; i < graph.num_nodes; i++)
	{
		states.label1[i] = false;
	}
	states.label1[arguments.sourceNode] = true;
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));


	Partitioner partitioner;

	timer.Start();

	uint gItr = 0;

	bool finished;
	bool* d_finished;
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));

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
				cout << "\t\tIteration " << ++itr << endl;
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				sswp_async << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (partitioner.partitionNodeSize[i],
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

			cout << itr << ((itr > 1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i << endl;
		}

		subgen.generate(graph, states, subgraph);

	}

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime / 1000 << " (s).\n";

	gpuErrorcheck(cudaMemcpy(states.value, states.d_value, graph.num_nodes * sizeof(uint), cudaMemcpyDeviceToHost));

	utilities::PrintResults(states.value, min(30, graph.num_nodes));

	if (arguments.hasOutput)
		utilities::SaveResults(arguments.output, states.value, graph.num_nodes);
}

