#include <cstdlib>

#include "../../shared/timer.hpp"


#include "../../shared/subgraph.cuh"
#include "../../shared/partitioner.cuh"
#include "../../shared/subgraph_generator.cuh"
#include "../../shared/gpu_error_check.cuh"
#include "../../shared/gpu_kernels.cuh"
#include "../../shared/subway_utilities.hpp"
#include "bc_dis.h"

void bc_bfs_compute(GraphStructure& graph, GraphStates<uint>& states, Subgraph& subgraph, SubgraphGenerator<uint>& subgen)
{
	Timer timer;
	timer.Start();

	unsigned int gItr = 0;

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
			cudaDeviceSynchronize();

			//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels << <partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (subgraph.d_activeNodes, states.d_label1, states.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);

			uint itr = 0;
			do
			{
				itr++;
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				bfs_async << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (partitioner.partitionNodeSize[i],
					partitioner.fromNode[i],
					partitioner.fromEdge[i],
					subgraph.d_activeNodes,
					subgraph.d_activeNodesPointer,
					subgraph.d_activeEdgeList,
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

bool bc_bfs_async(GraphStructure& graph, GraphStates<uint>& states, uint graph_value[], bool hasInit, int sourceNode)
{
	if (!hasInit)
	{
		for (uint i = 0; i < graph.num_nodes; i++)
		{
			states.value[i] = graph_value[i];
		}
		states.value[sourceNode] = 0;
		gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
		gpuErrorcheck(cudaMemcpy(states.d_value, states.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	}
	for (uint i = 0; i < graph.num_nodes; i++)
	{
		states.label1[i] = true;
		states.label2[i] = false;
	}
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	Subgraph subgraph(graph.num_nodes, graph.num_edges);
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

	bc_bfs_compute(graph, states, subgraph, subgen);

	subgraph.FreeSubgraph();
	subgen.FreeSubgraphGenerator();

	return false;
}

int find_max_distance(GraphStructure& graph, GraphStates<uint>& states)
{
	unsigned int level = 0;
	unsigned int *d_level;

	gpuErrorcheck(cudaMalloc(&d_level, sizeof(unsigned int)));
	cudaDeviceSynchronize();	
	gpuErrorcheck(cudaMemcpy(d_level, &level, sizeof(unsigned int), cudaMemcpyHostToDevice));
	find_max <<< graph.num_nodes/512 + 1 , 512 >>>(graph.num_nodes,
												states.d_value,
												d_level);	
	cudaDeviceSynchronize();
	gpuErrorcheck( cudaPeekAtLastError() );	
	gpuErrorcheck(cudaMemcpy(&level, d_level, sizeof(bool), cudaMemcpyDeviceToHost));
	level++;
	return level;
}

void bc_sigma_async(GraphStructure& graph, GraphStates<uint>& states, uint graph_sigma[], int level)
{
	for (uint i = 0; i < graph.num_nodes; i++)
	{
		states.sigma[i] = graph_sigma[i];
	}
	gpuErrorcheck(cudaMemcpy(states.d_sigma, states.sigma, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	if(level == -1) return;
	for (uint i = 0; i < graph.num_nodes; i++)
	{
		states.label1[i] = true;
		states.label2[i] = false;
	}
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	Subgraph subgraph(graph.num_nodes, graph.num_edges);
	SubgraphGenerator<uint> subgen(graph);
	subgen.generate(graph, states, subgraph);

	Partitioner partitioner;
	partitioner.partition(subgraph, subgraph.numActiveNodes);
	for(int i=0; i<partitioner.numPartitions; i++) {
		cudaDeviceSynchronize();
		gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
		
		bc_sigma_async <<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
												partitioner.fromNode[i],
												partitioner.fromEdge[i],
												subgraph.d_activeNodes,
												subgraph.d_activeNodesPointer,
												subgraph.d_activeEdgeList,
												graph.d_outDegree,
												states.d_value,
												states.d_sigma,
												level);	
		
		gpuErrorcheck( cudaDeviceSynchronize() );
		gpuErrorcheck( cudaPeekAtLastError() );
		
	}
	gpuErrorcheck(cudaMemcpy(states.sigma, states.d_sigma, graph.num_nodes * sizeof(uint), cudaMemcpyDeviceToHost));

	subgraph.FreeSubgraph();
	subgen.FreeSubgraphGenerator();
}

void bc_async(GraphStructure & graph, GraphStates<uint>&states, float graph_delta[], int level)
{
	for (uint i = 0; i < graph.num_nodes; i++)
	{
		states.delta[i] = graph_delta[i];
	}
	gpuErrorcheck(cudaMemcpy(states.d_delta, states.delta, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	//if(level == -1) return;
	for (uint i = 0; i < graph.num_nodes; i++)
	{
		states.label1[i] = true;
		states.label2[i] = false;
	}
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	Subgraph subgraph(graph.num_nodes, graph.num_edges);
	SubgraphGenerator<uint> subgen(graph);
	subgen.generate(graph, states, subgraph);

	Partitioner partitioner;
	partitioner.partition(subgraph, subgraph.numActiveNodes);
	for(int i=0; i<partitioner.numPartitions; i++) {
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));

			bc_ndp <<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
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
			
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
			
		}
	gpuErrorcheck(cudaMemcpy(states.delta, states.d_delta, graph.num_nodes * sizeof(float), cudaMemcpyDeviceToHost));

	subgraph.FreeSubgraph();
	subgen.FreeSubgraphGenerator();
}

