#include <iostream>
#include <mpi.h>

using std::cout;
using std::cerr;
using std::endl;

#include "cc_sig.h"
#include "cc_dis.h"

int main(int argc, char* argv[])
{
	ArgumentParser arguments(argc, argv, true, false);
	// cpu
	if (arguments.deviceID == -1)
	{
		std::string cmd = "./Components " + arguments.input;
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
	if (!arguments.isDistributed)//����
	{
		if (!arguments.isSynchronize)//�첽
		{
			cout << "single-async!" << endl;
			cc_sig_async(arguments);
		}
		else//ͬ��
		{
			cout << "single-sync!" << endl;
			cc_sig_sync(arguments);
		}
	}
	else//�ֲ�ʽ
	{
		if (!arguments.isSynchronize)//�첽
		{
			cout << "dist-async!" << endl;
			cc_dis_async(arguments);
		}
		else//ͬ��
		{
			cout << "error! not support dist-sync!" << endl;
		}
	}
	return 0;
}
