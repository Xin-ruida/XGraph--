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
	if (!arguments.isDistributed)//����
	{
		if (!arguments.isSynchronize)//�첽
		{
			cout << "single-async!" << endl;
			sswp_sig_async(arguments);
		}
		else//ͬ��
		{
			cout << "single-sync!" << endl;
			sswp_sig_sync(arguments);
		}
	}
	else//�ֲ�ʽ
	{
		if (!arguments.isSynchronize)//�첽
		{
			cout << "dist-async!" << endl;
			sswp_dis_async(arguments);
		}
		else//ͬ��
		{
			cout << "error! not support dist-sync!" << endl;
		}
	}
	return 0;
}
