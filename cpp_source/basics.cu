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
	int *shape = new int[4];
	shape[0] = batches; shape[1] = maps; shape[2] = rows; shape[3] = cols;
	if(split_axis==-1){ return shape; }

	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	int size = shape[split_axis];
	int split_size = 1+ (size/gpus);
	int split_offsize = size - ((gpus-1)*split_size);
	if(size % gpus == 0)
	{
		split_size -=1;
		split_offsize =split_size;
	}

	if(size == gpus){split_offsize = 1; split_size = 1;}
	if(gpuidx==gpus-1){shape[split_axis] = split_offsize; }
	else{shape[split_axis] = split_size;}

	assert(split_size*(gpus-1)+split_offsize >= gpus);

	return shape;

}

//go around export "C" with this declaration
template <typename T>
TensorTemplate<T>* empty_template(int batches, int maps, int rows, int cols, int split_axis)
{

	TensorTemplate<T> *out = new TensorTemplate<T>();
	int size = batches*maps*rows*cols;
	size_t bytes = size*sizeof(T);
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
		out->bytes_gpus.push_back(shape[0]*shape[1]*shape[2]*shape[3]*sizeof(T));

		T *gpu_data;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&gpu_data, out->bytes_gpus.back()));

		if(i == 0){ out->data = gpu_data; }
		out->data_gpus.push_back(gpu_data);
	}

	CUDA_CHECK_RETURN(cudaSetDevice(0));

	return out;
}

Tensor *empty_like(Tensor *A){ return empty(A->batches, A->maps, A->rows, A->cols, A->splitAxis); }
Tensor *empty(int batches, int maps, int rows, int cols){ return empty(batches, maps, rows, cols, -1); }
Tensor *empty(int batches, int maps, int rows, int cols, int split_axis)
{ return (Tensor*)empty_template<float>(batches, maps, rows, cols, split_axis); }

CharTensor *empty_char_like(Tensor *A){ return empty_char(A->batches, A->maps, A->rows, A->cols, A->splitAxis); }
CharTensor *empty_char(int batches, int maps, int rows, int cols){ return empty_char(batches, maps, rows, cols, -1); }
CharTensor *empty_char(int batches, int maps, int rows, int cols, int split_axis)
{ return (CharTensor*)empty_template<unsigned char>(batches, maps, rows, cols, split_axis); }


UIntTensor *empty_uint_like(Tensor *A){ return empty_uint(A->batches, A->maps, A->rows, A->cols/32, A->splitAxis); }
UIntTensor *empty_uint(int batches, int maps, int rows, int cols){ return empty_uint(batches, maps, rows, cols, -1); }
UIntTensor *empty_uint(int batches, int maps, int rows, int cols, int split_axis)
{ return (UIntTensor*)empty_template<unsigned int>(batches, maps, rows, cols, split_axis); }

UShortTensor *empty_ushort_like(Tensor *A){ return empty_ushort(A->batches, A->maps, A->rows, A->cols, A->splitAxis); }
UShortTensor *empty_ushort(int batches, int maps, int rows, int cols){ return empty_ushort(batches, maps, rows, cols, -1); }
UShortTensor *empty_ushort(int batches, int maps, int rows, int cols, int split_axis)
{ return (UShortTensor*)empty_template<unsigned int>(batches, maps, rows, cols, split_axis); }

void slice_axis(Tensor *A, Tensor *out)
{
	//only row slice supported right now
	assert(out->splitAxis == 2 && A->splitAxis == -1);
	Slice *S = emptySlice();
	S->batch_stop = A->batches;
	S->map_stop = A->maps;
	S->col_stop = A->cols;
	S->row_stop = 0;

	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		S->row_stop += out->shape_gpus[i][2];
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		//this is a complete mess, an evil monster, but will do for now
		kSlice<<<dim3(A->shape_gpus[i][0], A->shape_gpus[i][1],1),dim3(32,32,1)>>>(A->data_gpus[i],out->data_gpus[i],
				S->batch_start, S->batch_stop,
				S->map_start, S->map_stop,
				S->row_start, S->row_stop,
				S->col_start, S->col_stop,
				A->shape_gpus[i][2],A->shape_gpus[i][3],
				out->shape_gpus[i][0],out->shape_gpus[i][1],
				out->shape_gpus[i][3],out->shape_gpus[i][2], 1);

		CUDA_CHECK_RETURN(cudaPeekAtLastError());

		S->row_start += out->shape_gpus[i][2];
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

void stack_axis(Tensor *A, Tensor *out)
{
	//only row slice supported right now
		assert(out->splitAxis == -1 && A->splitAxis == 2);
		Slice *S = emptySlice();
		S->batch_stop = A->batches;
		S->map_stop = A->maps;
		S->col_stop = A->cols;
		S->row_stop = 0;

		int gpus = 0;
		CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
		for(int i = 0; i < gpus; i++)
		{
			S->row_stop = 0;
			S->row_start = 0;
			for(int j = 0; j < gpus; j++)
			{
				S->row_stop += A->shape_gpus[j][2];
				CUDA_CHECK_RETURN(cudaSetDevice(i));
				//this is a complete mess, an evil monster, but will do for now
					kSlice<<<dim3(out->shape_gpus[i][0], out->shape_gpus[i][1],1),dim3(32,32,1)>>>(A->data_gpus[j],out->data_gpus[i],
								S->batch_start, S->batch_stop,
								S->map_start, S->map_stop,
								S->row_start, S->row_stop,
								S->col_start, S->col_stop,
								out->shape_gpus[i][2],out->shape_gpus[i][3],
								A->shape_gpus[j][0],A->shape_gpus[j][1],
								A->shape_gpus[j][3],A->shape_gpus[j][2], 0);
				CUDA_CHECK_RETURN(cudaPeekAtLastError());

				S->row_start += A->shape_gpus[j][2];
			}
		}
		CUDA_CHECK_RETURN(cudaSetDevice(0));
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
	elementWise(out, NULL,NULL,0.0f,fill);
	return out;
}

Tensor *ones(int batches, int maps, int rows, int cols)
{
	Tensor *out = empty(batches,maps,rows,cols);
	elementWise(out, NULL,NULL,1.0f,fill);
	return out;
}


Tensor *T(Tensor *A)
{
	Tensor *out = empty(A->batches,A->maps,A->cols,A->rows);
	T(A,out, 2,3);
	out->rows = A->cols;
	out->cols = A->rows;
	return out;
}


void T(Tensor *A, Tensor *out,  int rows_idx, int cols_idx)
{
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		int rows = A->shape_gpus[i][rows_idx];
		int cols= A->shape_gpus[i][cols_idx];

		// setup execution parameters
		int grid_x = rows / COPY_BLOCK_SIZE;
		if (rows  % COPY_BLOCK_SIZE)
			grid_x++;

		int grid_y = cols / COPY_BLOCK_SIZE;
		if (cols % COPY_BLOCK_SIZE)
			grid_y++;

		dim3 grid(grid_x, grid_y, A->maps);
		dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kTransposeTensor<<< grid, threads >>>(A->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], rows, cols);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *to_col_major(Tensor *A)
{
  Tensor *out = empty_like(A);
  T(A, out, 3,2);

  return out;
}

void to_col_major(Tensor *A, Tensor *out)
{
	T(A, out, 3,2);
}

Tensor *to_row_major(Tensor *A)
{
	Tensor *out = empty_like(A);
	T(A, out, 2,3);

  return out;
}

void to_col_major_pinned(float  *A_data, float *out_data, int batches, int maps, int rows, int cols)
{
	unsigned int mapOffset = 0;
	unsigned int batchOffset = 0;

	//this is slow, but we usually do this to provide data to the batch allocator
	//which is in the right format, so nothing to worry about in terms of performance here
	for(int batch = 0; batch < batches; batch++ )
	{
		batchOffset = cols*rows*maps*batch;
		for(int map = 0; map < batches; map++ )
		{
			mapOffset = cols*rows*map;
			for(int row = 0; row < rows; row++)
			{
				for(int col = 0; col < cols; col++)
				{
					out_data[(col*rows) + row +batchOffset + mapOffset ] = A_data[(row*cols)+col + batchOffset + mapOffset];
				}
			}
		}
	}
}

Tensor *tocpu(Tensor *A, float *cpu_buffer)
{
	Tensor *temp = to_row_major(A);
	Tensor *out = new Tensor();

	CUDA_CHECK_RETURN(cudaMemcpy(cpu_buffer,temp->data_gpus[0],temp->bytes_gpus[0],cudaMemcpyDefault));
	out->batches = temp->batches;
	out->maps = temp->maps;
	out->rows = temp->rows;
	out->cols = temp->cols;
	out->bytes = temp->bytes;
	out->size = temp->size;
	out->data = cpu_buffer;
	out->isCUDA = 0;
	out->splitAxis = -1;

	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		CUDA_CHECK_RETURN(cudaFree(temp->data_gpus[i]));
	}


	CUDA_CHECK_RETURN(cudaSetDevice(0));
	delete temp;


	return out;
}

Tensor *tocpu(Tensor *A)
{
	Tensor *temp = to_row_major(A);
	Tensor *out = new Tensor();

	float *data = (float*)malloc(A->bytes);

	CUDA_CHECK_RETURN(cudaMemcpy(data,temp->data_gpus[0],temp->bytes_gpus[0],cudaMemcpyDefault));
	out->batches = temp->batches;
	out->maps = temp->maps;
	out->rows = temp->rows;
	out->cols = temp->cols;
	out->bytes = temp->bytes;
	out->size = temp->size;
	out->data = data;
	out->isCUDA = 0;
	out->splitAxis = -1;

	temp->freeTensor();


	return out;
}


void print_slice(Slice *S)
{
	cout << "batch: " << S->batch_start << " to " << S->batch_stop << endl;
	cout << "map: " << S->map_start << " to " << S->map_stop << endl;
	cout << "row: " << S->row_start << " to " << S->row_stop << endl;
	cout << "col: " << S->col_start << " to " << S->col_stop << endl;
}

void print_shape(int *shape)
{
	cout << shape[0] << "x" << shape[1] << "x" << shape[2]<< "x" << shape[3] << endl;
}

float print_free_memory()
{
	size_t total, free;
	cudaMemGetInfo(&free,&total);

	cout << "Free GB: " << ((float)free)/1024./1024./1024. << endl;

	return ((float)free)/1024./1024./1024.;
}

void print_tensor_shape(Tensor *A)
{
	for(int i = 0; i < A->data_gpus.size(); i++)
		print_shape(A->shape_gpus[i]);
}

void printsum(Tensor *A)
{
	cout << thrust_reduce(A, sum_tensor) << endl;
}


void printdim(Tensor *A)
{
	cout << A->rows << "x" << A->cols << endl;
}

void printmat(Tensor *A)
{
	Tensor * m = tocpu(A);
	print_cpu_matrix(m,0,A->rows, 0, A->cols);
	m->freeTensor();

}
void printmat(Tensor *A, int end_rows, int end_cols)
{
	Tensor *m = tocpu(A);
	print_cpu_matrix(m, 0,end_rows, 0,end_cols);
	m->freeTensor();

}

void printmat(Tensor *A, int start_row, int end_row, int start_col, int end_col)
{
	Tensor *m = tocpu(A);
	print_cpu_matrix(m, start_row, end_row, start_col, end_col);
	m->freeTensor();

}
void print_cpu_matrix(Tensor *A, int start_row, int end_row, int start_col, int end_col)
{

	int skip_rows = 0;
	int skip_cols = 0;
	if((end_row - start_row)*(end_col - start_col) > 1000)
	{
		if(end_row - start_row >= 20){ skip_rows = 3; }
		if(end_col - start_col >= 20){ skip_cols = 3; }
	}

	for(int row = start_row; row< end_row; row++)
	{
		if(row == skip_rows && skip_rows > 0){ printf("...,\n"); }
		if(skip_rows > 0 && (row >= skip_rows && row+3 < end_row)){ continue; }
		printf("[");
		for(int col =start_col; col < end_col; col++)
		{
			if(col == skip_cols && skip_cols > 0){ printf(" ..., "); }
			if(skip_cols > 0 && (col >= skip_cols && (col+3) < end_col)){ continue; }

		  if(A->data[(row*A->cols)+col] > 0.0f)
			  printf("% f ",A->data[(row*A->cols)+col]);
		  else
			  printf("%f ",A->data[(row*A->cols)+col]);
		}
		printf("]\n");
	}
	printf("\n");

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
			//print_shape(temp2->shape_gpus[i]);
			//cout << temp2->size_gpus[i] << endl;
			//cout << temp2->bytes_gpus[i] << endl;
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
	else if (split_axis == -1)
	{
		for(int i = 0; i < gpus; i++){ CUDA_CHECK_RETURN(cudaMemcpy(out->data_gpus[i],cpu_buffer,out->bytes_gpus[i],cudaMemcpyDefault)); }
		to_col_major(out, temp);
		for(int i = 0; i < gpus; i++){ CUDA_CHECK_RETURN(cudaMemcpy(out->data_gpus[i],temp->data_gpus[i],out->bytes_gpus[i],cudaMemcpyDefault)); }
	}
	else
	{
		throw "uden!";
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



Tensor *elementWise(Tensor *A, Tensor *B, Operation_t ops){ return elementWise(A, B, 0.0f, ops); }
Tensor *elementWise(Tensor *A, Tensor *B, float flt, Operation_t ops)
{ Tensor *out = empty_like(A); elementWise(A, B, out, flt, ops); return out; }
void elementWise(Tensor *A, Tensor *B, Tensor *out, Operation_t ops){ elementWise(A,B,out, 0.0f, ops); }
void elementWise(Tensor *A, Tensor *B, Tensor *out, float flt, Operation_t ops)
{
		int gpus = 0;
		CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
		for(int i = 0; i < gpus; i++)
		{
			int block_size = (A->shape_gpus[i][2]*A->shape_gpus[i][3]/THREADS_PER_BLOCKS) + 1;
			CUDA_CHECK_RETURN(cudaSetDevice(i));
			if(B && out)
				kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i],A->size_gpus[i], flt, ops);
			else if(out)
				kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, out->data_gpus[i],A->size_gpus[i], flt, ops);
			else
				kElementWise<<<block_size,THREADS_PER_BLOCKS>>>(A->data_gpus[i], NULL, NULL,A->size_gpus[i], flt, ops);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			if(ops == print){ CUDA_CHECK_RETURN(cudaDeviceSynchronize());}
		}
		CUDA_CHECK_RETURN(cudaSetDevice(0));

}
Tensor *vectorWise(Tensor *A, Tensor *B, Operation_t ops){ return vectorWise(A,B,0.0f,ops); }
Tensor *vectorWise(Tensor *A, Tensor *B, float flt, Operation_t ops)
{ Tensor *out = empty_like(A); vectorWise(A, B, out, flt, ops);	return out; }
void vectorWise(Tensor *A, Tensor *B, Tensor *out, Operation_t ops){ vectorWise(A,B,out,0.0f,ops); }
void vectorWise(Tensor *A, Tensor *B, Tensor *out, float flt, Operation_t ops)
{
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		int block_size = (A->shape_gpus[i][2]*A->shape_gpus[i][3]/THREADS_PER_BLOCKS) + 1;
		dim3 grid(block_size, A->shape_gpus[i][1],A->shape_gpus[i][0]);
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kVectorWise<<<grid,THREADS_PER_BLOCKS>>>(A->data_gpus[i], B->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][0], A->shape_gpus[i][2], A->shape_gpus[i][3]*A->shape_gpus[i][2], ops);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}


Tensor *softmax(Tensor *A){ Tensor *out = empty_like(A); softmax(A,out); return out; }
void softmax(Tensor *A, Tensor *out)
{
	//dim3 grids(A->batches, A->maps);
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		int blocks = A->shape_gpus[i][2];
		int threads = 1024;
		if(512 > A->shape_gpus[i][3]) threads = 512;
		if(256 > A->shape_gpus[i][3]) threads = 256;
		if(128 > A->shape_gpus[i][3]) threads = 128;
		if(64 > A->shape_gpus[i][3]) threads = 64;
		if(32 > A->shape_gpus[i][3]) threads = 32;
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		kSoftMax<<<blocks,threads,threads*2*sizeof(float)>>>(A->data_gpus[i], out->data_gpus[i], A->shape_gpus[i][2], A->shape_gpus[i][3]);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *argmax(Tensor *A){ Tensor *out = empty(A->batches,A->maps,A->rows,1, A->splitAxis); argmax(A,out); return out; }
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




float thrust_reduce(Tensor *A, Operation_t strategy)
{
	float value = 0;
	if(A->splitAxis == -1)
	{
		thrust::device_ptr<float> ptr(A->data);
		switch(strategy)
		{
			case sum_tensor: value = thrust::reduce(ptr, ptr+A->size); break;
			case max_tensor: value = thrust::reduce(ptr, ptr+A->size,-1.0f, thrust::maximum<float>()); break;
			case min_tensor: value = thrust::reduce(ptr, ptr+A->size,-1.0f, thrust::minimum<float>()); break;
		}

	}
	else
	{
		switch(strategy){ case max_tensor: value = -FLT_MAX; break; case min_tensor: value = FLT_MAX; break; }
		int gpus = 0;
		CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
		for(int i = 0; i < gpus; i++)
		{
			CUDA_CHECK_RETURN(cudaSetDevice(i));
			thrust::device_ptr<float> ptr(A->data_gpus[i]);
			switch(strategy)
			{
				case sum_tensor: value += thrust::reduce(ptr, ptr+A->size_gpus[i]); break;
				case max_tensor: value = fmax(value,thrust::reduce(ptr, ptr+A->size_gpus[i],-1.0f, thrust::maximum<float>())); break;
				case min_tensor: value = fmin(value,thrust::reduce(ptr, ptr+A->size_gpus[i],-1.0f, thrust::minimum<float>())); break;
			}
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
		}
		CUDA_CHECK_RETURN(cudaSetDevice(0));
	}

	return value;
}


void compression(Tensor *tbl_flt, CharTensor *charT, UIntTensor *uintT, UShortTensor *ushortT, Tensor *uncompressed, float precision, Tensor *errors, Tensor *avgPos, Tensor *avgNeg, Compression_t strategy)
{
	int gpus = 0;
		CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
		for(int i = 0; i < gpus; i++)
		{
			int blocks = (uncompressed->size_gpus[i]/THREADS_PER_BLOCKS) + 1;
			CUDA_CHECK_RETURN(cudaSetDevice(i));
			switch(strategy)
			{
				case compress_1bit: kCompression_1bit<<<blocks,THREADS_PER_BLOCKS>>>(uncompressed->data_gpus[i], errors->data_gpus[i],avgPos->data_gpus[i], avgNeg->data_gpus[i], uintT->data_gpus[i],uncompressed->shape_gpus[i][2],uncompressed->shape_gpus[i][3]); break;
				case compress_8bit: kCompression_8bit<<<blocks,THREADS_PER_BLOCKS>>>(tbl_flt->data_gpus[i], uncompressed->data_gpus[i], precision, uncompressed->size_gpus[i], charT->data_gpus[i]); break;
				case compress_16bit: kCompression_16bit<<<blocks,THREADS_PER_BLOCKS>>>(uncompressed->data_gpus[i], ushortT->data_gpus[i], uncompressed->size_gpus[i]); break;
				case decompress_1bit: kDecompression_1bit<<<blocks,THREADS_PER_BLOCKS>>>(uintT->data_gpus[i], errors->data_gpus[i],avgPos->data_gpus[i], avgNeg->data_gpus[i], uncompressed->data_gpus[i],uncompressed->shape_gpus[i][2],uncompressed->shape_gpus[i][3]); break;
				case decompress_8bit: kDecompression_8bit<<<blocks,THREADS_PER_BLOCKS>>>(tbl_flt->data_gpus[i],  charT->data_gpus[i], precision, charT->size_gpus[i], uncompressed->data_gpus[i]); break;
				case decompress_16bit: kDecompression_16bit<<<blocks,THREADS_PER_BLOCKS>>>(ushortT->data_gpus[i], uncompressed->data_gpus[i], ushortT->size_gpus[i]); break;
				default:
					throw "Not supported!";
			}

			CUDA_CHECK_RETURN(cudaPeekAtLastError());
		}
		CUDA_CHECK_RETURN(cudaSetDevice(0));
}

void compression_8bit(Tensor *tbl_flt, Tensor *A, float precision,  CharTensor *out)
{ compression(tbl_flt, out, 0, 0, A, precision, 0,0,0, compress_8bit); }
void decompression_8bit(Tensor *tbl_flt, CharTensor *A, float precision,  Tensor *out)
{ compression(tbl_flt, A, 0, 0, out, precision, 0,0,0, decompress_8bit); }
void compression_1bit(Tensor *A_with_errors, Tensor *errors, Tensor *avgPos, Tensor *avgNeg, UIntTensor *out)
{ compression(0, 0, out, 0, A_with_errors, 0.0f, errors,avgPos,avgNeg, compress_1bit); }
void decompression_1bit(UIntTensor *quant, Tensor *errors, Tensor *avgPos, Tensor *avgNeg, Tensor *out)
{ compression(0, 0, quant, 0, out, 0.0f, errors,avgPos,avgNeg, decompress_1bit); }
void compression_16bit(Tensor *A, UShortTensor *out)
{ compression(0, 0, 0, out, A, 0.0f, 0,0,0, compress_16bit); }
void decompression_16bit(UShortTensor *A, Tensor *out)
{ compression(0, 0, 0, A, out, 0.0f, 0,0,0, decompress_16bit); }

void reduceRow(Tensor *A, Tensor *out_values, Tensor *out_idxes, RowReduction_t strategy)
{
	int gpus = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&gpus));
	for(int i = 0; i < gpus; i++)
	{
		int blocks = min(256,(A->shape_gpus[i][2]));
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		if(out_idxes)
			kReduceRow<<<blocks,32>>>(A->data_gpus[i], out_values->data_gpus[i], out_idxes->data_gpus[i], A->rows, A->cols, strategy);
		else
			kReduceRow<<<blocks,32>>>(A->data_gpus[i], out_values->data_gpus[i], NULL, A->rows, A->cols, strategy);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));

}

cudaEvent_t* tick()
{
    cudaEvent_t* startstop;
    startstop = (cudaEvent_t*)malloc(2*sizeof(cudaEvent_t));
    cudaEventCreate(&startstop[0]);
    cudaEventCreate(&startstop[1]);
    cudaEventRecord(startstop[0], 0);

    return startstop;
}
float tock(cudaEvent_t* startstop){ return tock(startstop, "Time for the kernel(s): "); }
float tock(cudaEvent_t* startstop, std::string text)
{
	float time;
	cudaEventRecord(startstop[1], 0);
	cudaEventSynchronize(startstop[1]);
	cudaEventElapsedTime(&time, startstop[0], startstop[1]);
	printf((text + ": %f ms.\n").c_str(), time);
	return time;
}
float tock(std::string text, float tocks)
{
	printf((text + ": %f ms.\n").c_str(), tocks);
	return tocks;
}
float tock(cudaEvent_t* startstop, float tocks)
{
	float time;
	cudaEventRecord(startstop[1], 0);
	cudaEventSynchronize(startstop[1]);
	cudaEventElapsedTime(&time, startstop[0], startstop[1]);

	return time+tocks;
}
