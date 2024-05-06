#include <cstdlib>

#include "../shared/globals.hpp"
#include "../shared/timer.hpp"
#include "../shared/argument_parsing.cuh"
#include "../shared/graph.cuh"
#include "../shared/subgraph.cuh"
#include "../shared/partitioner.cuh"
#include "../shared/subgraph_generator.cuh"
#include "../shared/gpu_error_check.cuh"
#include "../shared/gpu_kernels.cuh"
#include "../shared/subway_utilities.hpp"


__global__ void bc_w(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							unsigned int *sigma,
							float *bc,
							bool *finished,
							int level)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(dist[id] != level)
			return;
			
		unsigned int sourceWeight = dist[id];

		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			
			finalDist = sourceWeight + 1;
			if(finalDist < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , level + 1);

				*finished = false;
			}
			if(dist[edgeList[i].end] == finalDist ) {
				atomicAdd(&sigma[edgeList[i].end] , sigma[id]);
			}
		}
	}
}


int main(int argc, char** argv)
{
	
	cudaFree(0);

	ArgumentParser arguments(argc, argv, true, false);
	
	Timer timer;
	timer.Start();
	
	GraphStructure graph;
	graph.ReadGraph(arguments.input);
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
	
	GraphStates<uint> states(graph.num_nodes, true, true, true);
	
	//cout<<11<<endl;
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		states.value[i] = DIST_INFINITY;
		states.sigma[i] = 0;
		states.delta[i] = 0;
		states.label1[i] = true;
		states.label2[i] = false;
	}
	//graph.value[arguments.sourceNode] = 0;
	//graph.label[arguments.sourceNode] = true;
	states.value[arguments.sourceNode] = 0;
	states.sigma[arguments.sourceNode] = 1;
	states.delta[arguments.sourceNode] = 0;


	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_value, states.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_sigma, states.sigma, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_delta,states.delta, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	
	Subgraph subgraph(graph.num_nodes, graph.num_edges);
	
	SubgraphGenerator<uint> subgen(graph);
	
	subgen.generate(graph, states, subgraph);	
	for(unsigned int i=0; i<graph.num_nodes; i++)//仅将源顶点标记为活跃顶点
	{
		states.label1[i] = false;
	}
	states.label1[arguments.sourceNode] = true;

	Partitioner partitioner;
	
	timer.Start();
	
	uint gItr = 0;
	uint level = 0;
	bool finished;
	bool *d_finished;
	bool all_finished;
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	partitioner.partition(subgraph, subgraph.numActiveNodes);

	do
	{
		all_finished = true;
		uint itr = 0;//分区迭代数
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			//cudaDeviceSynchronize();

			//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			//mixLabels<<<partitioner.partitionNodeSize[i]/512 + 1 , 512>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
				
			bc_w <<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
													partitioner.fromNode[i],
													partitioner.fromEdge[i],
													subgraph.d_activeNodes,
													subgraph.d_activeNodesPointer,
													subgraph.d_activeEdgeList,
													graph.d_outDegree,
													states.d_value,
													states.d_sigma,
													states.d_delta,
													d_finished,
													level);	
			cudaDeviceSynchronize();
			gpuErrorcheck( cudaPeekAtLastError() );	
				
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			if(!finished) all_finished = false;

		
		//subgen.generate(graph, subgraph);//根据活跃顶点重新生成子图
		}
		//cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << level << endl;
		level++;
	} while(!(all_finished));
	cout << level << endl;
	while(level > 1) {
		level--;
		for(int i=0; i<partitioner.numPartitions; i++) {

			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			bc <<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
													partitioner.fromNode[i],
													partitioner.fromEdge[i],
													subgraph.d_activeNodes,
													subgraph.d_activeNodesPointer,
													subgraph.d_activeEdgeList,
													graph.d_outDegree,
													states.d_value,
													states.d_sigma,
													states.d_delta,
													level);		
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
		}
		
	}

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";
	
	
	gpuErrorcheck(cudaMemcpy(states.value, states.d_value, graph.num_nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	gpuErrorcheck(cudaMemcpy(states.delta, states.d_delta, graph.num_nodes*sizeof(float), cudaMemcpyDeviceToHost));
	utilities::PrintResults(states.delta, min(100, graph.num_nodes));
	
		
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, states.delta, graph.num_nodes);
}

