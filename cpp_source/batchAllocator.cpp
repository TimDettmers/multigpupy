#include <batchAllocator.cuh>
#include <cmath>

BatchAllocator::BatchAllocator()
{
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&DEVICE_COUNT));
	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		cudaStream_t s;
		CUDA_CHECK_RETURN(cudaStreamCreate(&s));
		streams.push_back(s);
	}

	CUDA_CHECK_RETURN(cudaSetDevice(0));

	CUDA_CHECK_RETURN(cudaGetDeviceCount(&DEVICE_COUNT));
}



void BatchAllocator::allocateNextAsync(Tensor *batch, float *cpu_buffer)
{
	for(int i = 0; i < DEVICE_COUNT; i++)
		CUDA_CHECK_RETURN(cudaMemcpyAsync(batch->data_gpus[i],cpu_buffer,batch->bytes,cudaMemcpyDefault, streams[i]));
}

void BatchAllocator::replaceCurrentBatch(Tensor *current_batch, Tensor *next_batch)
{
	for(int i = 0; i < DEVICE_COUNT; i++)
		CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[i]));
}
