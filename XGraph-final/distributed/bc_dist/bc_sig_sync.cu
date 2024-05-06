#include "../../shared/timer.hpp"
#include "../../shared/subgraph.cuh"
#include "../../shared/partitioner.cuh"
#include "../../shared/subgraph_generator.cuh"
#include "../../shared/gpu_error_check.cuh"
#include "../../shared/gpu_kernels.cuh"
#include "../../shared/subway_utilities.hpp"
#include "bc_sig.h"

void bc_sig_sync(ArgumentParser arguments)
{
	
	cudaFree(0);

	
	Timer timer;
	timer.Start();
	
	GraphStructure graph;
	graph.ReadGraph(arguments.input);
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
	
	GraphStates<uint> states(graph.num_nodes, true, true, true);

	for (unsigned int i = 0; i < graph.num_nodes; i++)
	{
		states.value[i] = DIST_INFINITY;
		states.sigma[i] = 0;
		states.delta[i] = 0;
		states.label1[i] = false;
		states.label2[i] = false;
	}
	states.label2[arguments.sourceNode] = true;
	states.value[arguments.sourceNode] = 0;
	states.sigma[arguments.sourceNode] = 1;
	states.delta[arguments.sourceNode] = 0;


	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_value, states.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_sigma, states.sigma, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_delta, states.delta, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	
	Subgraph subgraph(graph.num_nodes, graph.num_edges);
	SubgraphGenerator<uint> subgen(graph);

	subgen.generate(graph, states, subgraph);

	Partitioner partitioner;

	timer.Start();
	
	uint gItr = 0;
	unsigned int level = 0;
	unsigned int *d_level;
	gpuErrorcheck(cudaMalloc(&d_level, sizeof(unsigned int)));
	while (subgraph.numActiveNodes > 0)
	{
		gItr++;
		
		partitioner.partition(subgraph, subgraph.numActiveNodes);
		// a super iteration
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();

			moveUpLabels << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (subgraph.d_activeNodes, states.d_label1, states.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);

			bc_kernel << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (partitioner.partitionNodeSize[i],
																			partitioner.fromNode[i],
																			partitioner.fromEdge[i],
																			subgraph.d_activeNodes,
																			subgraph.d_activeNodesPointer,
																			subgraph.d_activeEdgeList,
																			graph.d_outDegree,
																			states.d_value,
																			states.d_sigma,
																			states.d_delta,
																			states.d_label1,
																			states.d_label2);
			cudaDeviceSynchronize();
			gpuErrorcheck( cudaPeekAtLastError() );	

			//cout << "Global Iteration " << gItr << ", Partition " << i  << ", has active nodes " << subgraph.numActiveNodes << endl;
		}
		subgen.generate(graph, states, subgraph);		
	}	

	cudaDeviceSynchronize();
	gpuErrorcheck(cudaMemcpy(d_level, &level, sizeof(unsigned int), cudaMemcpyHostToDevice));
	find_max << < graph.num_nodes / 512 + 1, 512 >> > (graph.num_nodes,
		states.d_value,
		d_level);
	cudaDeviceSynchronize();
	gpuErrorcheck(cudaPeekAtLastError());
	gpuErrorcheck(cudaMemcpy(&level, d_level, sizeof(bool), cudaMemcpyDeviceToHost));
	level++;
	cout<< level << endl;

	for (unsigned int i = 0; i < graph.num_nodes; i++)
	{
		states.label1[i] = true;
		states.label2[i] = false;
	}
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	subgen.generate(graph, states, subgraph);

	while (level >= 1) {
		level--;
		//cout<< level << "  " <<partitioner1.numPartitions <<endl;
		partitioner.partition(subgraph, subgraph.numActiveNodes);
		for (int i = 0; i < partitioner.numPartitions; i++) {
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));

			bc_ndp << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (partitioner.partitionNodeSize[i],
																		partitioner.fromNode[i],
																		partitioner.fromEdge[i],
																		subgraph.d_activeNodes,
																		subgraph.d_activeNodesPointer,
																		subgraph.d_activeEdgeList,
																		graph.d_outDegree,
																		states.d_value,
																		states.d_sigma,
																		states.d_delta,
																		states.d_label1,
																		states.d_label2,
																		level);
			gpuErrorcheck(cudaPeekAtLastError());
			gpuErrorcheck(cudaDeviceSynchronize());
		}
		subgen.generate(graph, states, subgraph);
	}

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";
	
	gpuErrorcheck(cudaMemcpy(states.value, states.d_value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	gpuErrorcheck(cudaMemcpy(states.sigma, states.d_sigma, graph.num_nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	gpuErrorcheck(cudaMemcpy(states.delta, states.d_delta, graph.num_nodes*sizeof(float), cudaMemcpyDeviceToHost));
	utilities::PrintResults(states.delta, min(100, graph.num_nodes));
	
		
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, states.delta, graph.num_nodes);
}

