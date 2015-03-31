#include <Tensor.cuh>

void Tensor::freeTensor()
{
	if(onGPU)
	{
		for(int i = 0;i < data_gpus.size(); i++)
		{
			CUDA_CHECK_RETURN(cudaSetDevice(i));
			CUDA_CHECK_RETURN(cudaFree(data_gpus[i]));
		}
		CUDA_CHECK_RETURN(cudaSetDevice(0));
	}
	else{ free(data); }
	free(this);
}
