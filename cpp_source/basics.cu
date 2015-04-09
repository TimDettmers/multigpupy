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
	out->isCUDA = 1;

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

Tensor *empty_pinned(int batches, int maps, int rows, int cols, float *cpu_buffer)
{
	Tensor *out = new Tensor();
	int size = batches*maps*rows*cols;
	float *pinned_data;
	size_t bytes = size*sizeof(float);
	CUDA_CHECK_RETURN(cudaHostAlloc(&pinned_data, bytes, cudaHostAllocPortable));
	if(cpu_buffer)
		CUDA_CHECK_RETURN(cudaMemcpy(pinned_data,cpu_buffer,bytes,cudaMemcpyDefault));
	out->batches = batches;
	out->maps = maps;
	out->rows = rows;
	out->cols = cols;
	out->bytes = bytes;
	out->size = size;
	out->isCUDA = 1;
	out->data = pinned_data;

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
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
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
	out->isCUDA = 0;

	CUDA_CHECK_RETURN(cudaFree(temp->data));
	delete temp;


	return out;
}

void togpu(Tensor *out, float *cpu_buffer)
{

	Tensor *temp = empty(out->batches,out->maps,out->rows,out->cols);
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++){ CUDA_CHECK_RETURN(cudaMemcpy(out->data_gpus[i],cpu_buffer,out->bytes,cudaMemcpyDefault)); }
	to_col_major(out,temp);
	for(int i = 0; i < gpus; i++){ CUDA_CHECK_RETURN(cudaMemcpy(out->data_gpus[i],temp->data_gpus[i],out->bytes,cudaMemcpyDefault)); }

	temp->freeTensor();
}


Tensor *applySliceFunc(Tensor *A, Slice *S)
{
	Tensor *out = zeros(S->batch_stop-S->batch_start,
						S->map_stop-S->map_start,
						S->row_stop-S->row_start,
						S->col_stop-S->col_start);

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
		kSlice<<<dim3(A->batches, A->maps,1),dim3(32,32,1)>>>(A->data_gpus[i],out->data_gpus[i],
				S->batch_start, S->batch_stop,
				S->map_start, S->map_stop,
				S->row_start, S->row_stop,
				S->col_start, S->col_stop,
				A->rows,A->cols,out->batches,out->maps,out->cols,out->rows);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
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
			case copy: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size, flt, copy); break;
			case add_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size, flt, add_scalar); break;
			case mul_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size, flt, mul_scalar); break;
			case add_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size, flt, add_tensor); break;
			case sub_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size, flt, sub_tensor); break;
			case mul_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size, flt, mul_tensor); break;
			case div_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size, flt, div_tensor); break;
			case abs_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size, flt, abs_tensor); break;
			case log_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size, flt, log_tensor); break;
			case exp_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size, flt, exp_tensor); break;
			case pow_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size, flt, pow_tensor); break;
			case logistic: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size, flt, logistic); break;
			case logistic_grad: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size, flt, logistic_grad); break;
			case rectified_linear: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size, flt, rectified_linear); break;
			case eq_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size, flt,eq_tensor); break;
			case lt_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size, flt,lt_tensor); break;
			case gt_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size, flt,gt_tensor); break;
			case ge_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size, flt,ge_tensor); break;
			case le_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size, flt,le_tensor); break;
			case ne_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size, flt,ne_tensor); break;
			case eq_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size, flt,eq_scalar); break;
			case lt_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size, flt,lt_scalar); break;
			case gt_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size, flt,gt_scalar); break;
			case ge_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size, flt,ge_scalar); break;
			case le_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size, flt,le_scalar); break;
			case ne_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size, flt,ne_scalar); break;
			case dropout_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size, flt,dropout_tensor); break;
			case eq_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols, eq_vec); break;
			case lt_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols, lt_vec); break;
			case gt_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols, gt_vec); break;
			case le_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols, le_vec); break;
			case ge_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols, ge_vec); break;
			case ne_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols, ne_vec); break;
			case add_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols, add_vec); break;
			case sub_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols, sub_vec); break;
			case mul_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols, mul_vec); break;
			case div_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->batches, A->rows, A->rows*A->cols, div_vec); break;

			default: throw "Unsupported operation!";
		}
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}


void synchronize(Tensor *A, Tensor *out, int myid, int copyid, cudaStream_t stream,Operation_t ops)
{
	int block_size = (A->rows*A->cols/THREADS_PER_BLOCKS) + 1;
	CUDA_CHECK_RETURN(cudaSetDevice(myid));
	kElementWise<<<block_size,THREADS_PER_BLOCKS,0,stream>>>(A->data_gpus[myid],A->data_gpus[copyid],out->data_gpus[myid],A->size,0.0f,ops);
	CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


