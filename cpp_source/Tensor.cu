#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>

//this looks strange, but otherwise we cannot use template classes together with export "C" for ctypes

template <typename T>
class TensorTemplate
{
public:
	  int batches;
	  int maps;
	  int rows;
	  int cols;
	  int size;
	  size_t bytes;
	  T *data;
	  int isCUDA;
	  int splitAxis;
	  std::vector<int*> shape_gpus;
	  std::vector<int> size_gpus;
	  std::vector<size_t> bytes_gpus;
	  std::vector<T*> data_gpus;

void freeTensor()
{
	if(isCUDA)
	{
		for(int i = 0;i < data_gpus.size(); i++)
		{
			cudaSetDevice(i);
			cudaFree(data_gpus[i]);
		}
		cudaSetDevice(0);
	}
	else{ free(data); }
	free(this);
}



};
