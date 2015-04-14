#include <basics.cuh>
#include <assert.h>

using std::cout;
using std::endl;

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

int *get_split_shape(int batches, int maps, int rows, int cols,int split_axis,int gpuidx)
{
	int *ret = new int[4];
	ret[0] = batches; ret[1] = maps; ret[2] = rows; ret[3] = cols;
	if(split_axis==-1){ return ret; }

	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	int size = ret[split_axis];
	int split_size = 1+ (size/gpus);
	assert(split_size >= gpus);
	int split_offsize = size - ((gpus-1)*split_size);
	if(split_offsize == 0) split_offsize = split_size;

	if(split_size == gpus){split_offsize = 1; split_size = 1;}
	if(gpuidx==gpus-1){ret[split_axis] = split_offsize; }
	else{ret[split_axis] = split_size;}

	return ret;

}


Tensor *empty(int batches, int maps, int rows, int cols){ return empty(batches, maps, rows, cols, -1); }
Tensor *empty(int batches, int maps, int rows, int cols, int split_axis)
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
	out->splitAxis = split_axis;

	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		int *shape = get_split_shape(out->batches,out->maps, out->rows,out->cols, split_axis, i);

		out->shape_gpus.push_back(shape);
		out->size_gpus.push_back(shape[0]*shape[1]*shape[2]*shape[3]);
		out->bytes_gpus.push_back(shape[0]*shape[1]*shape[2]*shape[3]*sizeof(float));


		float *gpu_data;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&gpu_data, out->bytes_gpus.back()));

		if(i == 0){ out->data = gpu_data; }
		out->data_gpus.push_back(gpu_data);
	}

	CUDA_CHECK_RETURN(cudaSetDevice(0));

	return out;
}

float *empty_pinned(int batches, int maps, int rows, int cols, float *cpu_buffer)
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
	out->splitAxis = -1;
	out->data = pinned_data;

	return pinned_data;
}

Tensor *zeros(int batches, int maps, int rows, int cols){ return zeros(batches, maps, rows, cols, -1); }
Tensor *zeros(int batches, int maps, int rows, int cols, int split_axis)
{
	Tensor *out = empty(batches,maps,rows,cols,split_axis);
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
		thrust::fill(ptr_dev, ptr_dev + A->size_gpus[i],fill_value);
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
		kTransposeTensor<<< grid, threads >>>(A->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], rows, cols);
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
	out->splitAxis = -1;

	CUDA_CHECK_RETURN(cudaFree(temp->data));
	delete temp;


	return out;
}


void togpu(Tensor *out, float *cpu_buffer){ togpu(out, cpu_buffer, -1); }
void togpu(Tensor *out, float *cpu_buffer, int split_axis)
{
	Tensor *temp = empty(out->batches,out->maps,out->rows,out->cols);
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));

	if(split_axis==2)
	{
		for(int i = 0; i < gpus; i++){ CUDA_CHECK_RETURN(cudaMemcpy(temp->data_gpus[i],cpu_buffer,temp->bytes_gpus[i],cudaMemcpyDefault)); }
		Tensor *temp2 = to_col_major(temp);
		Slice *S = emptySlice();
		S->batch_stop = temp->batches;
		S->map_stop = temp->maps;
		S->col_stop = temp->cols;
		S->row_stop = 0;
		for(int i = 0; i < gpus; i++)
		{
			S->row_stop += out->shape_gpus[i][2];
			CUDA_CHECK_RETURN(cudaSetDevice(i));
			kSlice<<<dim3(temp2->shape_gpus[i][0], temp2->shape_gpus[i][1],1),dim3(32,32,1)>>>(temp2->data_gpus[i],out->data_gpus[i],
					S->batch_start, S->batch_stop,
					S->map_start, S->map_stop,
					S->row_start, S->row_stop,
					S->col_start, S->col_stop,
					temp2->shape_gpus[i][2],temp2->shape_gpus[i][3],
					out->shape_gpus[i][0],out->shape_gpus[i][1],
					out->shape_gpus[i][3],out->shape_gpus[i][2], 1);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());

			S->row_start += out->shape_gpus[i][2];
		}
		temp2->freeTensor();
	}
	if(split_axis == -1)
	{
		for(int i = 0; i < gpus; i++){ CUDA_CHECK_RETURN(cudaMemcpy(out->data_gpus[i],cpu_buffer,out->bytes_gpus[i],cudaMemcpyDefault)); }
		to_col_major(out, temp);
		for(int i = 0; i < gpus; i++){ CUDA_CHECK_RETURN(cudaMemcpy(out->data_gpus[i],temp->data_gpus[i],out->bytes_gpus[i],cudaMemcpyDefault)); }
	}

	CUDA_CHECK_RETURN(cudaSetDevice(0));
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
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kSlice<<<dim3(A->shape_gpus[i][0], A->shape_gpus[i][1],1),dim3(32,32,1)>>>(A->data_gpus[i],out->data_gpus[i],
				S->batch_start, S->batch_stop,
				S->map_start, S->map_stop,
				S->row_start, S->row_stop,
				S->col_start, S->col_stop,
				A->shape_gpus[i][2],A->shape_gpus[i][3],
				out->shape_gpus[i][0],out->shape_gpus[i][1],
				out->shape_gpus[i][3],out->shape_gpus[i][2],1);
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
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		int block_size = (A->shape_gpus[i][2]*A->shape_gpus[i][3]/THREADS_PER_BLOCKS) + 1;
		dim3 grid(block_size, A->shape_gpus[i][1],A->shape_gpus[i][0]);
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		switch(ops)
		{
			case copy: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size_gpus[i], flt, copy); break;
			case add_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size_gpus[i], flt, add_scalar); break;
			case mul_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size_gpus[i], flt, mul_scalar); break;
			case add_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size_gpus[i], flt, add_tensor); break;
			case sub_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size_gpus[i], flt, sub_tensor); break;
			case mul_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size_gpus[i], flt, mul_tensor); break;
			case div_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->size_gpus[i], flt, div_tensor); break;
			case abs_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size_gpus[i], flt, abs_tensor); break;
			case log_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size_gpus[i], flt, log_tensor); break;
			case exp_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size_gpus[i], flt, exp_tensor); break;
			case pow_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size_gpus[i], flt, pow_tensor); break;
			case logistic: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size_gpus[i], flt, logistic); break;
			case logistic_grad: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size_gpus[i], flt, logistic_grad); break;
			case rectified_linear: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i], A->size_gpus[i], flt, rectified_linear); break;
			case eq_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size_gpus[i], flt,eq_tensor); break;
			case lt_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size_gpus[i], flt,lt_tensor); break;
			case gt_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size_gpus[i], flt,gt_tensor); break;
			case ge_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size_gpus[i], flt,ge_tensor); break;
			case le_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size_gpus[i], flt,le_tensor); break;
			case ne_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size_gpus[i], flt,ne_tensor); break;
			case eq_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size_gpus[i], flt,eq_scalar); break;
			case lt_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size_gpus[i], flt,lt_scalar); break;
			case gt_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size_gpus[i], flt,gt_scalar); break;
			case ge_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size_gpus[i], flt,ge_scalar); break;
			case le_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size_gpus[i], flt,le_scalar); break;
			case ne_scalar: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size_gpus[i], flt,ne_scalar); break;
			case dropout_tensor: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size_gpus[i], flt,dropout_tensor); break;
			case eq_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], A->shape_gpus[i][2], A->shape_gpus[i][3]*A->shape_gpus[i][2], eq_vec); break;
			case lt_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], A->shape_gpus[i][2], A->shape_gpus[i][3]*A->shape_gpus[i][2], lt_vec); break;
			case gt_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], A->shape_gpus[i][2], A->shape_gpus[i][3]*A->shape_gpus[i][2], gt_vec); break;
			case le_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], A->shape_gpus[i][2], A->shape_gpus[i][3]*A->shape_gpus[i][2], le_vec); break;
			case ge_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], A->shape_gpus[i][2], A->shape_gpus[i][3]*A->shape_gpus[i][2], ge_vec); break;
			case ne_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], A->shape_gpus[i][2], A->shape_gpus[i][3]*A->shape_gpus[i][2], ne_vec); break;
			case add_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], A->shape_gpus[i][2], A->shape_gpus[i][3]*A->shape_gpus[i][2], add_vec); break;
			case sub_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], A->shape_gpus[i][2], A->shape_gpus[i][3]*A->shape_gpus[i][2], sub_vec); break;
			case mul_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], A->shape_gpus[i][2], A->shape_gpus[i][3]*A->shape_gpus[i][2], mul_vec); break;
			case div_vec: kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], A->shape_gpus[i][2], A->shape_gpus[i][3]*A->shape_gpus[i][2], div_vec); break;
			case print: kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, NULL,A->size_gpus[i], flt,print); break;

			default: throw "Unsupported operation!";
		}
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		if(ops == print){ CUDA_CHECK_RETURN(cudaDeviceSynchronize());}
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}


Tensor *softmax(Tensor *A){ Tensor *out = empty(A->batches,A->maps,A->rows,A->cols); softmax(A,out); return out; }
void softmax(Tensor *A, Tensor *out)
{
	dim3 grids(A->batches, A->maps);
	dim3 threads(A->rows > THREADS_PER_BLOCKS ? THREADS_PER_BLOCKS : A->rows, 1);
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kSoftMax<<<grids,threads >>>(A->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][2], A->shape_gpus[i][3]);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *argmax(Tensor *A){ Tensor *out = empty(A->batches,A->maps,A->rows,1); argmax(A,out); return out; }
void argmax(Tensor *A, Tensor *out)
{
	dim3 grids(A->batches, A->maps);
	dim3 threads(A->rows > THREADS_PER_BLOCKS ? THREADS_PER_BLOCKS : A->rows, 1);
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kArgmax<<<grids,threads >>>(A->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][2], A->shape_gpus[i][3]);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

void weightUpdate(Tensor *RMS, Tensor *grad, float RMS_multiplier, float learning_rate, int batch_size, weightUpdate_t strategy)
{

	int blocks = (RMS->size/THREADS_PER_BLOCKS) + 1;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kWeightUpdate<<<blocks,THREADS_PER_BLOCKS>>>(RMS->data_gpus[i], grad->data_gpus[i], RMS_multiplier, learning_rate, batch_size, RMS->size, strategy);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}


void synchronize(Tensor *A, Tensor *out, int myid, int copyid, cudaStream_t stream,Operation_t ops)
{
	int block_size = (A->shape_gpus[myid][2]*A->shape_gpus[myid][3]/THREADS_PER_BLOCKS) + 1;
	CUDA_CHECK_RETURN(cudaSetDevice(myid));
	kElementWise<<<block_size,THREADS_PER_BLOCKS,0,stream>>>(A->data_gpus[myid],A->data_gpus[copyid],out->data_gpus[myid],A->size_gpus[myid],0.0f,ops);
	//TODO: add vstack method with backward slice
	CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

float sum(Tensor *A)
{
	thrust::device_ptr<float> ptr(A->data);
	return thrust::reduce(ptr, ptr+A->size);
}

float max(Tensor *A)
{
	thrust::device_ptr<float> ptr(A->data);
	float res = -1.0f;
	return thrust::reduce(ptr, ptr+A->size,res, thrust::maximum<float>());
}

float min(Tensor *A)
{
	thrust::device_ptr<float> ptr(A->data);
	float res = -1.0f;
	return thrust::reduce(ptr, ptr+A->size,res, thrust::minimum<float>());
}

void synchronizingStack(Tensor* A, Tensor *out)
{
	float **h_arrA = new float*[A->data_gpus.size()];
	for (int i = 0; i < A->data_gpus.size(); i++)
		h_arrA[i] = A->data_gpus[i];

	float **d_arrA;
	cudaMalloc((void**) &d_arrA, sizeof(float*)*A->data_gpus.size());
	cudaMemcpy(d_arrA, h_arrA, sizeof(float*)*A->data_gpus.size(),cudaMemcpyDefault);

	dim3 griddim(out->cols, A->data_gpus.size()) ;
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	int cumulative_size = 0;
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		cumulative_size += out->size_gpus[i];
		vStackN<<<griddim,((out->rows/A->data_gpus.size()) > 512 ? 512 : (out->rows/A->data_gpus.size()))>>>(d_arrA, out->data_gpus[i], out->shape_gpus[i][3], A->shape_gpus[0][2], A->shape_gpus.back()[2]);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));



	free(h_arrA);
	cudaFree(d_arrA);
}
