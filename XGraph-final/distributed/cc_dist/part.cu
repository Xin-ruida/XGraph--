#include "cc_dis.h"
#include "../../shared/gpu_error_check.cuh"
void part(GraphStructure graph, GraphStructure graph_cut[], GraphStates<uint> states, GraphStates<uint> states_cut[], uint n)
{
    uint nodes = graph.num_nodes;
    uint edges = graph.num_edges;
    // locality-aware chunking
    uint *partition_offset = new uint [n+1]{};
    partition_offset[0] = 0;
    uint alpha = (n - 1) * 8;
    uint remained_amount = edges  + nodes * alpha;
    for (uint i=0;i<n;i++)
    {
        uint remained_partitions = n - i;
        uint expected_chunk_size = remained_amount / remained_partitions;
        if (remained_partitions == 1)
        {
            partition_offset[i+1] = nodes;
        }else{
            uint got_edges = 0;
            for(uint j=partition_offset[i];j<nodes;j++)
            {
                got_edges += graph.outDegree[j] + alpha;

                if (got_edges > expected_chunk_size)
                {
                    partition_offset[i+1] = j;
                    break;
                }
            }
            // partition_offset[i+1] = partition_offset[i+1] / Pagesize * Pagesize;
        }
        for (uint v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++)
        {
            remained_amount -= graph.outDegree[v_i] + alpha;
        }
    }
    uint flag=0;
    for(uint i=0;i<n+1;i++)
    {
        // cout << partition_offset[i] << " ";
        if(i>1&&partition_offset[i]==0)
        {
            flag = 1;
        }
    }
    cout << endl;

    uint partition_len[n] = {0};
    for(uint i=0;i<n;i++)
    {
        partition_len[i] = partition_offset[i+1] - partition_offset[i];
    }

    // locality-aware chunking useless, simple cut
    if(flag==1)
    {
        uint step_len = graph.num_nodes/n;
        
        for(uint i=0;i<n;i++)
        {
            partition_len[i] = step_len;
        }
        partition_len[n-1] = graph.num_nodes - step_len*(n-1);
        
        partition_offset[0] = 0;
        for(uint i=1;i<n;i++)
        {
            partition_offset[i] += step_len;
        }
        partition_offset[n] = graph.num_nodes;
        // for(uint i=0;i<n+1;i++)
        // {
        //     cout << partition_offset[i] << " ";
        // }
        // cout << endl;
    }
    
   // master nodes csr
    //uint mirror_max_node[n]={0};
    uint p=0;
    for(uint i=0;i<n;i++)
    {
        graph_cut[i].num_nodes = partition_offset[i+1];
        graph_cut[i].num_edges = 0;
        uint tn = partition_len[i];
        for(uint j=p;j<p+tn;j++)
        {
            graph_cut[i].num_edges+=graph.outDegree[j];
        }
        uint te = graph_cut[i].num_edges;
               
        graph_cut[i].edgeList = new OutEdge[te];
        for(uint j=0; j<te; j++)
		{
			graph_cut[i].edgeList[j].end = graph.edgeList[graph.nodePointer[p]+j].end;
            //mirror_max_node[i] = mirror_max_node[i] > graph_cut[i].edgeList[j].end ? mirror_max_node[i]:graph_cut[i].edgeList[j].end;
		}
        if (states.edgeWeight != NULL)
        {
            states_cut[i].edgeWeight = new uint[te];
            for (uint j = 0; j < te; j++)
            {
                states_cut[i].edgeWeight[j] = states.edgeWeight[graph.nodePointer[p] + j];
            }
        }

        // mirror nodes
        //graph_cut[i].num_nodes = max(partition_offset[i+1], mirror_max_node[i]+1);
        graph_cut[i].num_nodes = graph.num_nodes;
        graph_cut[i].nodePointer = new uint[graph_cut[i].num_nodes]{0};
        for(uint j=0; j<tn; j++)
        {
            graph_cut[i].nodePointer[p+j] = graph.nodePointer[p+j] - graph.nodePointer[p];
        }
        if(graph_cut[i].num_nodes != tn)
            for(uint j= p + tn;j<graph_cut[i].num_nodes;j++)
            {
                graph_cut[i].nodePointer[j] = graph_cut[i].nodePointer[p + tn-1] + graph.outDegree[p + tn-1];
            }

        p = p + tn;
    }
}
