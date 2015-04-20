#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

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
		int gpus = 0;
		CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
		for(int i = 0;i < gpus; i++)
		{
			CUDA_CHECK_RETURN(cudaSetDevice(i));
			CUDA_CHECK_RETURN(cudaFree(data_gpus[i]));
		}
		CUDA_CHECK_RETURN(cudaSetDevice(0));
	}
	else{ free(data); }
	free(this);
}



};
