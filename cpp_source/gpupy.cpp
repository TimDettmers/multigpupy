#include <gpupy.cuh>
#include <basics.cuh>
#include <cudaKernels.cuh>

GPUpy::GPUpy(){}

void GPUpy::init(int seed)
{
	DEVICE_COUNT = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&DEVICE_COUNT));


	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		//CUDA_CHECK_RETURN(cudaDeviceReset());
		curandGenerator_t gen;

		CURAND_CHECK_RETURN(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A));
		CURAND_CHECK_RETURN(curandSetPseudoRandomGeneratorSeed(gen, seed));
		CURAND_CHECK_RETURN(curandSetGeneratorOffset(gen, 100));

		generators.push_back(gen);
	}

	cudaSetDevice(0);

}


Tensor *GPUpy::rand(int batchsize, int mapsize, int rows, int cols)
{ Tensor *out = empty(batchsize, mapsize, rows, cols); rand(out); return out; }
void GPUpy::rand(Tensor *out)
{
	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		CURAND_CHECK_RETURN(curandGenerateUniform(generators[i], out->data_gpus[i],out->size));
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *GPUpy::normal(int batchsize, int mapsize, int rows, int cols, float mean, float std)
{ Tensor *out = empty(batchsize, mapsize, rows, cols); normal(mean, std, out); return out; }
void GPUpy::normal(float mean, float std, Tensor *out)
{
	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		CURAND_CHECK_RETURN(curandGenerateNormal(generators[i], out->data_gpus[i],out->size,mean,std));
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *GPUpy::randn(int batchsize, int mapsize, int rows, int cols){ return normal(batchsize,mapsize, rows, cols, 0.0f,1.0f); }
void GPUpy::randn(Tensor *out){ normal(0.0f,1.0f,out); }




