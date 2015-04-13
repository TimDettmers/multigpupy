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

		cudaStream_t s;
		CUDA_CHECK_RETURN(cudaStreamCreate(&s));
		streams.push_back(s);
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
		CURAND_CHECK_RETURN(curandGenerateUniform(generators[i], out->data_gpus[i],out->size_gpus[i]));
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
		CURAND_CHECK_RETURN(curandGenerateNormal(generators[i], out->data_gpus[i],out->size_gpus[i],mean,std));
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
void GPUpy::Tdot(Tensor *A, Tensor *B, Tensor *out){ dot(A,B,out,CUBLAS_OP_T,CUBLAS_OP_N); }
void GPUpy::dot(Tensor *A, Tensor *B, Tensor *out, cublasOperation_t T1, cublasOperation_t T2)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		int A_rows = A->shape_gpus[i][2], A_cols = A->shape_gpus[i][3], B_cols = B->shape_gpus[i][3];
		if (T1 == CUBLAS_OP_T)
		{
			A_rows = A->shape_gpus[i][3];
			A_cols = A->shape_gpus[i][2];
		}
		if (T2 == CUBLAS_OP_T)
			B_cols = B->shape_gpus[i][2];

		assert(A->shape_gpus[i][1] == 1 && "Tensors dot product is not supported.");
		assert(A->shape_gpus[i][0] == 1 && "Tensors dot product is not supported.");


		CUDA_CHECK_RETURN(cudaSetDevice(i));

		CUBLAS_CHECK_RETURN(cublasSgemm(cublashandles[i], T1, T2, A_rows, B_cols,
				A_cols, &alpha, A->data_gpus[i], A->shape_gpus[i][2], B->data_gpus[i], B->shape_gpus[i][2], &beta,
				out->data_gpus[i], out->shape_gpus[i][2]));
	}

	CUDA_CHECK_RETURN(cudaSetDevice(0));


}


Tensor *GPUpy::dropout(Tensor *A, float dropout_rate)
{
	Tensor *out = empty(A->batches, A->maps, A->rows, A->cols);

	dropout(A, out, dropout_rate);
	return out;
}

void GPUpy::dropout(Tensor *A, Tensor *out, float dropout_rate)
{
	rand(out);
	applyFunc(A, NULL, out, dropout_rate, dropout_tensor);
}



void GPUpy::enablePeerAccess()
{
	for(int gpu1 = 0; gpu1 < DEVICE_COUNT; gpu1++)
		for(int gpu2 = 0; gpu2 < DEVICE_COUNT; gpu2++)
			if(gpu1!=gpu2)
			{
				CUDA_CHECK_RETURN(cudaSetDevice(gpu1));
				CUDA_CHECK_RETURN(cudaDeviceEnablePeerAccess(gpu2,0));
			}

	hasPeerAccess = true;
}

Tensor *GPUpy::synchronizingAdd(Tensor *A){ Tensor *out = empty(A->batches,A->maps,A->rows,A->cols); synchronizingAdd(A,out); return out; }
void GPUpy::synchronizingAdd(Tensor *A, Tensor *out)
{
	if(!hasPeerAccess){ enablePeerAccess(); }

	int copyid = 0;
	for(int offset = 1; offset < DEVICE_COUNT; offset++)
	{
		for(int myid = 0; myid < DEVICE_COUNT; myid++)
		{
			copyid = myid + offset;
			copyid = copyid >= DEVICE_COUNT ? copyid-DEVICE_COUNT : copyid;

			synchronize(A,out,myid,copyid,streams[myid], add_tensor);

		}
		for(int myid = 0; myid < DEVICE_COUNT; myid++){ CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[myid]));}
	}

}

