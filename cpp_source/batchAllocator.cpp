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
		cudaStream_t s_y;
		CUDA_CHECK_RETURN(cudaStreamCreate(&s_y));
		streams_y.push_back(s_y);
	}

	int isEnabled = 0;
	cudaDeviceCanAccessPeer(&isEnabled, 0,1);
	if(isEnabled==0)
		for(int gpu1 = 0; gpu1 < DEVICE_COUNT; gpu1++)
			for(int gpu2 = 0; gpu2 < DEVICE_COUNT; gpu2++)
				if(gpu1!=gpu2)
				{
					CUDA_CHECK_RETURN(cudaSetDevice(gpu1));
					CUDA_CHECK_RETURN(cudaDeviceEnablePeerAccess(gpu2,0));
				}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}



void BatchAllocator::allocateNextAsync(Tensor *batch, float *cpu_buffer, Tensor *batch_y, float *cpu_buffer_y)
{
	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaMemcpyAsync(batch->data_gpus[i],cpu_buffer,batch->bytes,cudaMemcpyDefault, streams[i]));
		CUDA_CHECK_RETURN(cudaMemcpyAsync(batch_y->data_gpus[i],cpu_buffer_y,batch_y->bytes,cudaMemcpyDefault, streams_y[i]));
	}
}

void BatchAllocator::replaceCurrentBatch()
{


	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[i]));
		CUDA_CHECK_RETURN(cudaStreamSynchronize(streams_y[i]));
	}
}
