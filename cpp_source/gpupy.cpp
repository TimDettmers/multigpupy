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
		curandGenerator_t gen;

		CURAND_CHECK_RETURN(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A));
		CURAND_CHECK_RETURN(curandSetPseudoRandomGeneratorSeed(gen, seed));
		CURAND_CHECK_RETURN(curandSetGeneratorOffset(gen, 100));

		generators.push_back(gen);
		cublasHandle_t handle;
		CUBLAS_CHECK_RETURN(cublasCreate_v2(&handle));
		cublashandles.push_back(handle);
	}

	CUDA_CHECK_RETURN(cudaSetDevice(0));

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

Tensor *GPUpy::dot(Tensor *A, Tensor *B){ Tensor *out = empty(1,1,A->rows,B->cols); dot(A,B,out,CUBLAS_OP_N,CUBLAS_OP_N); return out; }
Tensor *GPUpy::Tdot(Tensor *A, Tensor *B){ Tensor *out = empty(1,1,A->rows,B->cols); dot(A,B,out,CUBLAS_OP_T,CUBLAS_OP_N); return out; }
Tensor *GPUpy::dotT(Tensor *A, Tensor *B){ Tensor *out = empty(1,1,A->rows,B->cols); dot(A,B,out,CUBLAS_OP_N,CUBLAS_OP_T); return out; }
void GPUpy::dot(Tensor *A, Tensor *B, Tensor *out){ dot(A,B,out,CUBLAS_OP_N,CUBLAS_OP_N); }
void GPUpy::dotT(Tensor *A, Tensor *B, Tensor *out){ dot(A,B,out,CUBLAS_OP_N,CUBLAS_OP_T); }
void GPUpy::TdotT(Tensor *A, Tensor *B, Tensor *out){ dot(A,B,out,CUBLAS_OP_T,CUBLAS_OP_N); }
void GPUpy::dot(Tensor *A, Tensor *B, Tensor *out, cublasOperation_t T1, cublasOperation_t T2)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	int A_rows = A->rows, A_cols = A->cols, B_cols = B->cols;
	if (T1 == CUBLAS_OP_T)
	{
		A_rows = A->cols;
		A_cols = A->rows;
	}
	if (T2 == CUBLAS_OP_T)
		B_cols = B->rows;

	assert(A->maps == 1 && "Tensors dot product is not supported.");
	assert(A->batches == 1 && "Tensors dot product is not supported.");


	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));

		CUBLAS_CHECK_RETURN(cublasSgemm(cublashandles[i], T1, T2, A_rows, B_cols,
				A_cols, &alpha, A->data_gpus[i], A->rows, B->data_gpus[i], B->rows, &beta,
				out->data_gpus[i], out->rows));
	}

	CUDA_CHECK_RETURN(cudaSetDevice(0));


}




