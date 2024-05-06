#include <iostream>
#include <mpi.h>

using std::cout;
using std::cerr;
using std::endl;

#include "bfs_sig.h"
#include "bfs_dis.h"

int main(int argc, char* argv[])
{
    ArgumentParser arguments(argc, argv, true, false);
	// cpu
	if (arguments.deviceID == -1)
	{
		std::string cmd = "./BFS " + arguments.input;
		FILE* pp = popen(cmd.data(), "r"); // build pipe
		if (!pp)
			return 1;
		// collect cmd execute result
		char tmp[1024];
		while (fgets(tmp, sizeof(tmp) * 1024, pp) != NULL)
			std::cout << tmp << std::endl; // can join each line as string
		pclose(pp);
		return 1;
	}
	//GPU
	if (!arguments.isDistributed)//单机
	{
		if (!arguments.isSynchronize)//异步
		{
			cout << "single-async!" << endl;
			bfs_sig_async(arguments);
		}
		else//同步
		{
			cout << "single-sync!" << endl;
			bfs_sig_sync(arguments);
		}
	}
	else//分布式
	{
		if (!arguments.isSynchronize)//异步
		{
			cout << "dist-async!" << endl;
			bfs_dis_async(arguments);
		}
		else//同步
		{
			cout << "error! not support dist-sync!" << endl;
		}
	}
    return 0;
}
