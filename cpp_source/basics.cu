#include <basics.cuh>



Tensor *empty(int batches, int maps, int rows, int cols)
{
	Tensor *out = new Tensor();
	int size = batches*maps*rows*cols;
	size_t bytes = size*sizeof(float);
	out->batches = batches;
	out->maps = maps;
	out->rows = rows;
	out->cols = cols;
	out->bytes = bytes;
	out->size = size;
	out->onGPU = 1;

	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		float *gpu_data;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&gpu_data, bytes));

		if(i == 0){ out->data = gpu_data; }
		out->data_gpus.push_back(gpu_data);
	}

	CUDA_CHECK_RETURN(cudaSetDevice(0));

	return out;
}

Tensor *zeros(int batches, int maps, int rows, int cols)
{
	Tensor *out = empty(batches,maps,rows,cols);
	return fill_with_number(out, 0.0f);
}

Tensor *ones(int batches, int maps, int rows, int cols)
{
	Tensor *out = empty(batches,maps,rows,cols);
	return fill_with_number(out, 1.0f);
}

Tensor *fill_with_number(Tensor *A, float fill_value)
{
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		thrust::device_ptr<float> ptr_dev(A->data_gpus[i]);
		thrust::fill(ptr_dev, ptr_dev + A->size,fill_value);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));

	return A;
}



Tensor *T(Tensor *A)
{
	Tensor *out = empty(A->batches,A->maps,A->cols,A->rows);
	T(A,out, A->rows,A->cols);
	out->rows = A->cols;
	out->cols = A->rows;
	return out;
}


void T(Tensor *A, Tensor *out,  int rows, int cols)
{
	// setup execution parameters
	int grid_x = rows / COPY_BLOCK_SIZE;
	if (rows  % COPY_BLOCK_SIZE)
		grid_x++;

	int grid_y = cols / COPY_BLOCK_SIZE;
	if (cols % COPY_BLOCK_SIZE)
		grid_y++;

	dim3 grid(grid_x, grid_y, A->maps);
	dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kTransposeTensor<<< grid, threads >>>(A->data_gpus[i], out->data_gpus[i], A->batches, rows, cols);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *to_col_major(Tensor *A)
{
  Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
  T(A, out, A->cols,A->rows);

  return out;
}

void to_col_major(Tensor *A, Tensor *out)
{
	T(A, out, A->cols,A->rows);
}

Tensor *to_row_major(Tensor *A)
{
	Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	T(A, out, A->rows,A->cols);

  return out;
}



Tensor *tocpu(Tensor *A, float *cpu_buffer)
{
	Tensor *temp = to_row_major(A);
	Tensor *out = new Tensor();

	CUDA_CHECK_RETURN(cudaMemcpy(cpu_buffer,temp->data,temp->bytes,cudaMemcpyDefault));
	out->batches = temp->batches;
	out->maps = temp->maps;
	out->rows = temp->rows;
	out->cols = temp->cols;
	out->bytes = temp->bytes;
	out->size = temp->size;
	out->data = cpu_buffer;
	out->onGPU = 0;

	CUDA_CHECK_RETURN(cudaFree(temp->data));
	delete temp;


	return out;
}

void togpu(Tensor *out, float *cpu_buffer)
{
	Tensor *temp = empty(out->batches,out->maps,out->rows,out->cols);
	CUDA_CHECK_RETURN(cudaMemcpy(out->data,cpu_buffer,out->bytes,cudaMemcpyDefault));
	to_col_major(out,temp);
	CUDA_CHECK_RETURN(cudaMemcpy(out->data,temp->data,out->bytes,cudaMemcpyDefault));
	CUDA_CHECK_RETURN(cudaFree(temp->data));
	free(temp);
}


void add(Tensor *A, Tensor *B, Tensor *out)
{
	//checkTensorOperation(A, B, out, CUBLAS_OP_N, CUBLAS_OP_N, 0);
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kAdd<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}


Tensor *add(Tensor *A, Tensor *B)
{
	Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	add(A, B, out);

	return out;
}

void sub(Tensor *A, Tensor *B, Tensor *out)
{
	//checkTensorOperation(A, B, out, CUBLAS_OP_N, CUBLAS_OP_N, 0);
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kSub<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}


Tensor *sub(Tensor *A, Tensor *B)
{
	Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	sub(A, B, out);

	return out;
}

void mul(Tensor *A, Tensor *B, Tensor *out)
{
	//checkTensorOperation(A, B, out, CUBLAS_OP_N, CUBLAS_OP_N, 0);
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kMul<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}


Tensor *mul(Tensor *A, Tensor *B)
{
	Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	mul(A, B, out);

	return out;
}

void div(Tensor *A, Tensor *B, Tensor *out)
{
	//checkTensorOperation(A, B, out, CUBLAS_OP_N, CUBLAS_OP_N, 0);
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kDiv<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}


Tensor *div(Tensor *A, Tensor *B)
{
	Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	div(A, B, out);

	return out;
}

Tensor *scalarMul(Tensor *A, float a)
{
	Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	scalarMul(A, a, out);

	return out;
}

void scalarMul(Tensor *A, float a, Tensor *out)
{
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kScalarMul<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], a, out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *scalarAdd(Tensor *A, float a)
{
	Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	scalarAdd(A, a, out);

	return out;
}

void scalarAdd(Tensor *A, float a, Tensor *out)
{
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kScalarAdd<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], a, out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *gpuExp(Tensor *A)
{
	  Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	  gpuExp(A, out);

	  return out;
}

void gpuExp(Tensor *A, Tensor *out)
{
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kExp<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *logistic(Tensor *A)
{
	  Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	  logistic(A, out);

	  return out;
}

void logistic(Tensor *A, Tensor *out)
{
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kLogistic<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *logisticGrad(Tensor *A)
{
	  Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	  logisticGrad(A, out);

	  return out;
}

void logisticGrad(Tensor *A, Tensor *out)
{
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kLogisticGrad<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *gpuLog(Tensor *A)
{
	  Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	  gpuLog(A, out);

	  return out;
}

void gpuLog(Tensor *A, Tensor *out)
{
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kLog<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *gpuSqrt(Tensor *A)
{
	  Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	  gpuSqrt(A, out);

	  return out;
}

void gpuSqrt(Tensor *A, Tensor *out)
{
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		 kSqrt<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *square(Tensor *A)
{
	Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	square(A, out);

  	return out;
}

void square(Tensor *A, Tensor *out)
{
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kSquare<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *abs(Tensor *A)
{
	Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	abs(A, out);

	return out;
}

void abs(Tensor *A, Tensor *out)
{
	int block_size = (A->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kAbs<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}




