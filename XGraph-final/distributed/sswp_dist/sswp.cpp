#include <iostream>
#include <mpi.h>

using std::cout;
using std::cerr;
using std::endl;

#include "sswp_sig.h"
#include "sswp_dis.h"

int main(int argc, char* argv[])
{
	ArgumentParser arguments(argc, argv, true, false);
	
	//GPU
	if (!arguments.isDistributed)//单机
	{
		if (!arguments.isSynchronize)//异步
		{
			cout << "single-async!" << endl;
			sswp_sig_async(arguments);
		}
		else//同步
		{
			cout << "single-sync!" << endl;
			sswp_sig_sync(arguments);
		}
	}
	else//分布式
	{
		if (!arguments.isSynchronize)//异步
		{
			cout << "dist-async!" << endl;
			sswp_dis_async(arguments);
		}
		else//同步
		{
			cout << "error! not support dist-sync!" << endl;
		}
	}
	return 0;
}
