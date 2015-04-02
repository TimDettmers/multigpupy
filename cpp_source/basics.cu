#include <basics.cuh>


Slice *emptySlice()
{
	Slice *out = new Slice();
	out->batch_start = 0;
	out->batch_stop = INT_MAX;
	out->map_start = 0;
	out->map_stop = INT_MAX;
	out->row_start = 0;
	out->row_stop = INT_MAX;
	out->col_start = 0;
	out->col_stop = INT_MAX;

	return out;
}

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


int sliceDimHelper(int dim, int start, int stop)
{
	if(start< 0 && stop == dim){ return -start; }
	if(start >= 0 && stop < 0){ return start+stop; }
	if(start >= 0 && stop<= dim){ return stop-start; }
	if(start == 0 && stop > dim){ return dim; }
	return 0;
}

Tensor *applySliceFunc(Tensor *A, Slice *S)
{
	Tensor *out = empty(
			sliceDimHelper(A->batches,S->batch_start,S->batch_stop),
			sliceDimHelper(A->maps,S->map_start,S->map_stop),
			sliceDimHelper(A->rows,S->row_start,S->row_stop),
			sliceDimHelper(A->cols,S->col_start,S->col_stop));
	applySliceFunc(A, S, out);

	return out;

}

void applySliceFunc(Tensor *A, Slice *S, Tensor *out)
{
	int block_size = (A->rows*A->cols/THREADS_PER_BLOCKS) + 1;
	dim3 grid(block_size, A->maps,A->batches);
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));

	}

	CUDA_CHECK_RETURN(cudaSetDevice(0));

}

Tensor *applyFunc(Tensor *A, Tensor *B, Operation_t ops){ return applyFunc(A,B,0.0f,ops); }
Tensor *applyFunc(Tensor *A, Tensor *B, float flt, Operation_t ops)
{
	Tensor *out = empty(A->batches,A->maps,A->rows,A->cols);
	applyFunc(A, B, out, flt, ops);

	return out;
}

void applyFunc(Tensor *A, Tensor *B, Tensor *out, Operation_t ops){ applyFunc(A,B,out,0.0f,ops); }
void applyFunc(Tensor *A, Tensor *B, Tensor *out, float flt, Operation_t ops)
{
	int block_size = (A->rows*A->cols/THREADS_PER_BLOCKS) + 1;
	dim3 grid(block_size, A->maps,A->batches);
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		switch(ops)
		{
			case add_scalar: kScalarAdd<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], flt, out->data_gpus[i], A->size); break;
			case mul_scalar: kScalarMul<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], flt, out->data_gpus[i], A->size); break;
			case add_tensor: kAdd<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size); break;
			case sub_tensor: kSub<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size); break;
			case mul_tensor: kMul<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size); break;
			case div_tensor: kDiv<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size); break;
			case add_vec: kAddVectorToTensor<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols); break;
			case sub_vec: kSubVectorToTensor<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols); break;
			case mul_vec: kMulVectorToTensor<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols); break;
			case div_vec: kDivVectorToTensor<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols); break;
			case abs_tensor: kAbs<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size); break;
			case log_tensor: kLog<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size); break;
			case exp_tensor: kExp<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size); break;
			case sqrt_tensor: kSqrt<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size); break;
			case pow_tensor: kPow<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], flt, out->data_gpus[i], A->size); break;
			case logistic: kLogistic<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size); break;
			case logistic_grad: kLogisticGrad<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], out->data_gpus[i], A->size); break;
			case eq_tensor: kCompare<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],eq_tensor, A->size); break;
			case ls_tensor: kCompare<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],ls_tensor, A->size); break;
			case gt_tensor: kCompare<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],gt_tensor, A->size); break;
			case ge_tensor: kCompare<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],ge_tensor, A->size); break;
			case le_tensor: kCompare<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],le_tensor, A->size); break;
			case ne_tensor: kCompare<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],ne_tensor, A->size); break;
			default: throw "Unsupported operation!";
		}
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}



