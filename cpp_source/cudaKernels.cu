#include <Tensor.cuh>

//const int NUM_THREADS = 32;


__global__ void kGetNonZeroElements(float *A, float *out, int size)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	for (unsigned int i = idx;i < size; i += numThreads)
		 atomicAdd(&out[0],A[i] != 0.0f ? 1.0f : 0.0f);
}

__global__ void kGetNonZeroColumns(float *A, float *out, int rows, int cols)
{
	const int myCol = (blockIdx.x * blockDim.x) + threadIdx.x;
	float result = 0.0f;

	if(myCol < cols)
	{
		for (unsigned int i = 0;i < rows; i++)
		{
			if(A[(myCol*rows) + i] != 0.0f)
				result = 1.0f;
		}

		atomicAdd(&out[0],result);
	}
}

__global__ void kRenormalizeWeights(float *w, float *unit_sums, float limit, int rows, int cols)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int size = rows*cols;

	int myCol = 0;
	float rel_diff = 0.0f;
	for (unsigned int i = idx;i < size; i += numThreads)
	{
		myCol = i/rows;
		if(unit_sums[myCol] > limit)
		{
			rel_diff = 1.0f/unit_sums[myCol];
			w[i] *= rel_diff;
		}
		else{ continue; }

	}

}

__global__ void kRdmNumbers(float *seed, int size, float *out)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned long long s[ 2 ];
	//s[0] = (long long)seed[(gridDim.x*blockIdx.x)  + threadIdx.x];
	//s[1] = (long long)seed[(gridDim.x*(blockIdx.x+1))  + threadIdx.x];

	s[0] = 17;
	s[1] = 83;
	unsigned long long s1 = s[ 0 ];
	unsigned long long s0 = s[ 1 ];
	unsigned long long rdm64 = 23459867034598355;


	if(idx == 0)
	{
		printf("rdm: %i\n", rdm64);
		printf("rdm1: %i\n", (unsigned int)(rdm64&0xffffffff));
		printf("rdm2: %i\n", (unsigned int)((rdm64>>32)&0xffffffff));
	}

    unsigned int rdm32_1 = 0;
    unsigned int rdm32_2 = 0;
	//printf("seed 1: %i\n", seed[(gridDim.x*blockIdx.x)  + threadIdx.x]);
	//printf("seed 2: %i\n", seed[(gridDim.x*(blockIdx.x+1))  + threadIdx.x]);
	//printf("idx: %i\n", idx);
	for(int i = idx*2; i < size; i+=numThreads*2)
	{
		s1 = s[0];
		s0 = s[1];
		s[0] = s0;
		s1 ^= s1 << 23; // a

		rdm64 =  (s[1 ] = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0; // b, c

		rdm32_1 = (rdm64&0xffffffff);
		rdm32_2 = ((rdm64>>32)&0xffffffff);
		out[i] = rdm32_1;
		out[i+1] = rdm32_2;

	}

	seed[(gridDim.x*blockIdx.x)  + threadIdx.x] = s[0];
	seed[(gridDim.x*(blockIdx.x+1))  + threadIdx.x] = s[1];

}

__global__ void kCreateRdmSqrtWeight_Logistic(float *A, int in, int out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const float lower_limit = -4.0f*sqrtf(6.0f/((float)in + out));
  const float upper_limit =  4.0f*sqrtf(6.0f/((float)in + out));
  const float range = upper_limit-lower_limit;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       A[i] = lower_limit + (A[i]*range);
  }
}

__global__ void kCreateSparseRdmWeight(float *rdm, float* indicies, float *out, int rows, int cols, int connections)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int connection_idx = 0;
  float rdm_value = 0.0f;
  int size = connections*cols;
  int current_col = 0;

  //each thread fills one row
  for (unsigned int i = idx; i < size; i += numThreads)
  {
	  connection_idx = (int)indicies[i];
	  rdm_value = rdm[i];
	  current_col = i/(connections);
	  out[(current_col*rows)+connection_idx] = rdm_value;
  }
}

__global__ void kRandInt(float *A, int lower_limit, int upper_limit, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int range = upper_limit-lower_limit + 1;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  //use uniform random sample to get integers
       A[i] = (float)(((int)((A[i]*range))) + lower_limit);
  }
}



//vertical stack for column major format
__global__ void vStack(float *A, float *B, float *out, int size_out, int rows_a, int rows, int cols)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  int current_col = 0;
  int current_row = 0;
  int offset = 0;
  const int rows_b = rows - rows_a;

  for (unsigned int i = idx;i < size_out; i += numThreads)
  {
	  current_col = i / rows; //int arithmetic
	  offset = (current_col*rows);
	  current_row = i - offset;

	  if(current_row >= rows_a)
	  {
		  //fetch b value
		  out[i] = B[(current_col*rows_b) + current_row - rows_a];
	  }
	  else
	  {
		  //fetch a value
		  out[i] = A[(current_col*rows_a) + current_row];
	  }
  }
}

__global__ void hStack(float *A, float *B, float *out, int size_out, int size_a)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for(unsigned int i = idx; i < size_out; i+=numThreads)
  {
	  if(i >= size_a)
	  {
		  //append B
		  out[i] = B[i - size_a];
	  }
	  else
	  {
		  //append A
		  out[i] = A[i];
	  }
  }

}

__global__ void hStackN(float **arrA, int general_size, float *out, int size_out, int matrices_count)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int current_matrix = 0;

  for(unsigned int i = idx; i < size_out; i+=numThreads)
  {
	  current_matrix = i / general_size;
	  current_matrix = current_matrix == matrices_count ? current_matrix - 1 : current_matrix;
	  out[i] = arrA[current_matrix][i - (current_matrix*general_size)];
  }

}

__global__ void vStackN(float **arrA, float *out, int full_rows, int block_rows, int block_off_rows)
{
  int myblockidx = 0;
  int myidx = 0;


  for(int block_row_idx = threadIdx.x; block_row_idx < block_rows; block_row_idx +=blockDim.x)
  {
	  if(block_row_idx > full_rows){continue;}
	  myblockidx =(block_rows*blockIdx.x)+block_row_idx;
	  myidx = (full_rows*blockIdx.x)+(blockIdx.y*block_rows)+block_row_idx;
	  out[myidx] = arrA[blockIdx.y][myblockidx];
  }

}

__global__ void AddGradientsN(float **arrA, int size, int myrank, int matrix_count, float multiplier)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	for(int matrix_idx = 0; matrix_idx < matrix_count; matrix_idx++)
	{
		if(matrix_idx == myrank){ continue; }

		for(unsigned int i = idx; i < size; i+=numThreads)
			arrA[myrank][i] += arrA[matrix_idx][i];
	}
	//better numerical stability to do it afterwards
	for(unsigned int i = idx; i < size; i+=numThreads)
		arrA[myrank][i] *=multiplier;

}

__global__ void kAdd_to_z(float *z, float *z1, float *y, float *y_count, int rows, int cols, float *out)
{
	float value = 0;
	for(int row = blockIdx.x; row < rows; row +=gridDim.x)
	{
		int cls = (int)y[row];
		if(threadIdx.x == 0)
			atomicAdd(&y_count[cls],1.0f);
		for (unsigned int col = threadIdx.x; col < cols; col += blockDim.x)
		{
			value = z1[row + (col*rows)];
			atomicAdd(&out[cls+(col*rows)],value);
		}
	}

	__syncthreads();

	for(int row = blockIdx.x; row < rows; row +=gridDim.x)
	{
		int cls = (int)y[row];
		for (unsigned int col = threadIdx.x; col < cols; col += blockDim.x)
		{
			if(y_count[cls] > 0)
				out[cls+(col*rows)] /= y_count[cls];
		}
	}

}


__global__ void kElementWise(float *A,float *B, float *out, int size, float flt, Operation_t strategy)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	switch(strategy)
	{
		  case copy: for (unsigned int i = idx;i < size; i += numThreads) out[i] = A[i]; break;
	  	  case add_scalar: for (unsigned int i = idx;i < size; i += numThreads) out[i] = A[i] + flt; break;
	  	  case mul_scalar: for (unsigned int i = idx;i < size; i += numThreads) out[i] = A[i] * flt; break;
		  case add_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = A[i] + B[i]; break;
		  case sub_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = A[i] - B[i]; break;
		  case mul_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = A[i] * B[i]; break;
		  case div_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = B[i] == 0.0 ? 0.0 : fdividef(A[i], B[i]); break;
		  case exp_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = __expf(A[i]); break;
		  case log_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = __logf(A[i]); break;
		  case abs_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = fabs(A[i]); break;
		  case logistic: for (unsigned int i = idx;i < size; i += numThreads) out[i] = __fdividef(1.0f , (1.0 + __expf(-A[i]))); break;
		  case logistic_grad: for (unsigned int i = idx;i < size; i += numThreads) out[i] = A[i]*(1.0f-A[i]); break;
		  case rectified_linear: for (unsigned int i = idx;i < size; i += numThreads) out[i] = A[i] > 0.0f ? A[i] : 0.0f; break;
		  case double_rectified_linear: for (unsigned int i = idx;i < size; i += numThreads) out[i] = A[i] > 0.0f && A[i] < 1.0f ? A[i] : 0.0f; break;
		  case double_rectified_linear_grad: for (unsigned int i = idx;i < size; i += numThreads) out[i] = A[i] > 0.0f && A[i] < 1.0f ? 1.0f : 0.0f; break;
		  case eq_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] == B[i]); break;
		  case lt_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] < B[i]); break;
		  case gt_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] > B[i]); break;
		  case le_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] <= B[i]); break;
		  case ge_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] >= B[i]); break;
		  case ne_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] != B[i]); break;
		  case eq_scalar: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] == flt); break;
		  case lt_scalar: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] < flt); break;
		  case gt_scalar: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] > flt); break;
		  case le_scalar: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] <= flt); break;
		  case ge_scalar: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] >= flt); break;
		  case ne_scalar: for (unsigned int i = idx;i < size; i += numThreads) out[i] = (float)(A[i] != flt); break;
		  case dropout_tensor: for (unsigned int i = idx;i < size; i += numThreads) out[i] = out[i] > flt ? A[i] : 0.0f; break;
		  case print: for (unsigned int i = idx;i < size; i += numThreads) printf("%f ",A[i]); break;
		  case fill:  for (unsigned int i = idx;i < size; i += numThreads) A[i] = flt; break;
		  case pow_tensor:
			  int flt_int = flt;
			  if(flt == (float)flt_int && flt_int % 2 == 0)
				  for (unsigned int i = idx;i < size; i += numThreads)
					  out[i] = A[i] < 0.0f ? __powf(-A[i],flt) : __powf(A[i],flt);
			  else if(flt == (float)flt_int)
				  for (unsigned int i = idx;i < size; i += numThreads)
					  out[i] = A[i] < 0.0f ? -__powf(-A[i],flt) : __powf(A[i],flt);
			  else
				  for (unsigned int i = idx;i < size; i += numThreads)
					  out[i] = __powf(A[i],flt);


		  	  break;
	}

}

__global__ void kVectorWise(float *A, float *v, float *out, int batches, int rows, int size, Operation_t strategy)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int mapOffset = size*blockIdx.y;
	unsigned int batchOffset = size*gridDim.y*blockIdx.z;
	int offset = 0;

	switch(strategy)
	{
		case add_vec:
			for (unsigned int i = idx;i < size; i += numThreads)
			{ offset = (i / rows); out[i + batchOffset + mapOffset] =  A[i + batchOffset + mapOffset] + v[offset]; } break;
		case sub_vec:
			for (unsigned int i = idx;i < size; i += numThreads)
			{ offset = (i / rows); out[i + batchOffset + mapOffset] =  A[i + batchOffset + mapOffset] - v[offset]; } break;
		case mul_vec:
			for (unsigned int i = idx;i < size; i += numThreads)
			{ offset = (i / rows); out[i + batchOffset + mapOffset] =  A[i + batchOffset + mapOffset] * v[offset]; } break;
		case div_vec:
			for (unsigned int i = idx;i < size; i += numThreads)
			{ offset = (i / rows); out[i + batchOffset + mapOffset] =  A[i + batchOffset + mapOffset] / v[offset]; } break;
		case eq_vec:
			for (unsigned int i = idx;i < size; i += numThreads)
			{ offset = (i / rows); out[i + batchOffset + mapOffset] =  (float)(A[i + batchOffset + mapOffset] == v[offset]); } break;
		case ne_vec:
			for (unsigned int i = idx;i < size; i += numThreads)
			{ offset = (i / rows); out[i + batchOffset + mapOffset] =  (float)(A[i + batchOffset + mapOffset] != v[offset]); } break;
		case gt_vec:
			for (unsigned int i = idx;i < size; i += numThreads)
			{ offset = (i / rows); out[i + batchOffset + mapOffset] =  (float)(A[i + batchOffset + mapOffset] > v[offset]); } break;
		case lt_vec:
			for (unsigned int i = idx;i < size; i += numThreads)
			{ offset = (i / rows); out[i + batchOffset + mapOffset] =  (float)(A[i + batchOffset + mapOffset] < v[offset]); } break;
		case le_vec:
			for (unsigned int i = idx;i < size; i += numThreads)
			{ offset = (i / rows); out[i + batchOffset + mapOffset] =  (float)(A[i + batchOffset + mapOffset] <= v[offset]); } break;
		case ge_vec:
			for (unsigned int i = idx;i < size; i += numThreads)
			{ offset = (i / rows); out[i + batchOffset + mapOffset] =  (float)(A[i + batchOffset + mapOffset] >= v[offset]); } break;
	}
}

__global__ void kSub_Sparse(float *A, float *data, int *ptr_rows, int *idx_cols, float *out, int rows, int cols, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int row_idx = 0;

  for (unsigned int i = idx;i < rows*cols; i += numThreads)
	  out[i] = A[i];

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  for(int j = 0; j < rows + 1; j++)
	  {
		  if(ptr_rows[j] > i)
		  {
			  row_idx = j-1;
			  break;
		  }
	  }
      out[(idx_cols[i] * rows) + row_idx] = A[(idx_cols[i] * rows) + row_idx] - data[i];
  }
}
 
__global__ void kTranspose(float *A, float *out, int width, int height) 
{
    __shared__ float block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

    // read the Matrix *tile into shared memory
    unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < width) && (yIndex < height)) 
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = A[index_in];
    }

    __syncthreads();

    // write the transposed Matrix *tile to global memory
    xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < height) && (yIndex < width)) 
    {
        unsigned int index_out = yIndex * height + xIndex;
        out[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

__global__ void kTransposeTensor(float *A, float *out, int batches, int width, int height)
{
    __shared__ float block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

    // read the Matrix *tile into shared memory
    unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;
    unsigned int mapOffset = width*height*blockIdx.z;
    unsigned int batchOffset = width*height*gridDim.z;

	for(int batch = 0; batch < batches; batch++ )
	{
	    xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
	    yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;
		if((xIndex < width) && (yIndex < height))
		{
			unsigned int index_in = (yIndex * width) + xIndex + mapOffset + (batch*batchOffset);
			block[threadIdx.y][threadIdx.x] = A[index_in];
		}

		__syncthreads();

		// write the transposed Matrix *tile to global memory
		xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
		yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

		if((xIndex < height) && (yIndex < width))
		{
			unsigned int index_out = yIndex * height + xIndex + mapOffset + (batch*batchOffset);
			out[index_out] = block[threadIdx.x][threadIdx.y];
		}
	}

}


__device__ void reduceToMax(float* sdata, unsigned int tid, int threads)
{

  //Synchronize threads to share shared memory data
  __syncthreads();

  float myMax = sdata[tid];

  // do reduction in shared mem
  if (threads >= 512) { if (tid < 256) { sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 256]); } __syncthreads(); }
  if (threads >= 256) { if (tid < 128) { sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 128]); } __syncthreads(); }
  if (threads >= 128) { if (tid <  64) { sdata[tid] = myMax = fmaxf(myMax, sdata[tid +  64]); } __syncthreads(); }

  if (threads == 32){
    if (tid < 16)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (threads >=  32) { smem[tid] = myMax = fmaxf(myMax, smem[tid + 16]); }
      if (threads >=  16) { smem[tid] = myMax = fmaxf(myMax, smem[tid +  8]); }
      if (threads >=   8) { smem[tid] = myMax = fmaxf(myMax, smem[tid +  4]); }
      if (threads >=   4) { smem[tid] = myMax = fmaxf(myMax, smem[tid +  2]); }
      if (threads >=   2) { smem[tid] = myMax = fmaxf(myMax, smem[tid +  1]); }
    }
  }
  else
  {
    if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (threads >=  64) { smem[tid] = myMax = fmaxf(myMax, smem[tid + 32]); }
      if (threads >=  32) { smem[tid] = myMax = fmaxf(myMax, smem[tid + 16]); }
      if (threads >=  16) { smem[tid] = myMax = fmaxf(myMax, smem[tid +  8]); }
      if (threads >=   8) { smem[tid] = myMax = fmaxf(myMax, smem[tid +  4]); }
      if (threads >=   4) { smem[tid] = myMax = fmaxf(myMax, smem[tid +  2]); }
      if (threads >=   2) { smem[tid] = myMax = fmaxf(myMax, smem[tid +  1]); }
    }
  }
}

__device__ void reduceToMaxAndArgMax(float* sdataMax, float* sdataArgMax, unsigned int tid, int threads)
{
  	float myMax = sdataMax[tid];
  	if(threads == 32)
  	{
		if (tid < 16)
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile float* smemMax = sdataMax;
			volatile float* smemArgMax = sdataArgMax;
			if (threads >=  32) if(myMax < smemMax[tid + 16]){smemMax[tid] = myMax = smemMax[tid + 16];  smemArgMax[tid] = smemArgMax[tid + 16]; }
			if (threads >=  16) if(myMax < smemMax[tid +  8]){smemMax[tid] = myMax = smemMax[tid +  8];  smemArgMax[tid] = smemArgMax[tid +  8]; }
			if (threads >=   8) if(myMax < smemMax[tid +  4]){smemMax[tid] = myMax = smemMax[tid +  4];  smemArgMax[tid] = smemArgMax[tid +  4]; }
			if (threads >=   4) if(myMax < smemMax[tid +  2]){smemMax[tid] = myMax = smemMax[tid +  2];  smemArgMax[tid] = smemArgMax[tid +  2]; }
			if (threads >=   2) if(myMax < smemMax[tid +  1]){smemMax[tid] = myMax = smemMax[tid +  1];  smemArgMax[tid] = smemArgMax[tid +  1]; }
		}
  	}
	else
	{
		if (tid < 32)
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile float* smemMax = sdataMax;
			volatile float* smemArgMax = sdataArgMax;
			if (threads >=  64) if(myMax < smemMax[tid + 32]){smemMax[tid] = myMax = smemMax[tid + 32];  smemArgMax[tid] = smemArgMax[tid + 32]; }
			if (threads >=  32) if(myMax < smemMax[tid + 16]){smemMax[tid] = myMax = smemMax[tid + 16];  smemArgMax[tid] = smemArgMax[tid + 16]; }
			if (threads >=  16) if(myMax < smemMax[tid +  8]){smemMax[tid] = myMax = smemMax[tid +  8];  smemArgMax[tid] = smemArgMax[tid +  8]; }
			if (threads >=   8) if(myMax < smemMax[tid +  4]){smemMax[tid] = myMax = smemMax[tid +  4];  smemArgMax[tid] = smemArgMax[tid +  4]; }
			if (threads >=   4) if(myMax < smemMax[tid +  2]){smemMax[tid] = myMax = smemMax[tid +  2];  smemArgMax[tid] = smemArgMax[tid +  2]; }
			if (threads >=   2) if(myMax < smemMax[tid +  1]){smemMax[tid] = myMax = smemMax[tid +  1];  smemArgMax[tid] = smemArgMax[tid +  1]; }
		}

	}
}

template <int strategy, int isVolatile>
__device__ float reduce_value(float aggregate, float *sdata, volatile float *smem, float *sdata_idx, volatile float *smem_idx, unsigned int tid, unsigned int offset)
{

	if(isVolatile)
	{
		switch(strategy)
		{
			case row_sum:
				smem[tid] = aggregate = aggregate + sdata[tid + offset];
				return aggregate;
			case row_max:
				smem[tid] = aggregate = fmaxf(aggregate,sdata[tid + offset]);
				return aggregate;
			case row_argmax:
				if(aggregate <= smem[tid + offset])
				{
					sdata_idx[tid] = sdata_idx[tid + offset];
					sdata[tid] = aggregate = sdata[tid + offset];
				}
				return aggregate;
			default: return NAN;
		}
	}
	else
	{
		switch(strategy)
		{
			case row_sum:
				smem[tid] = aggregate = aggregate + smem[tid + offset];
				return aggregate;
			case row_max:
				smem[tid] = aggregate = fmaxf(aggregate,smem[tid + offset]);
				return aggregate;
			case row_argmax:
				if(aggregate <= smem[tid + offset])
				{
					smem_idx[tid] = smem_idx[tid + offset];
					smem[tid] = aggregate = smem[tid + offset];
				}
				return aggregate;
			default: return NAN;
		}
	}

}

template <int strategy>
__device__ void reduce(float* sdata, float *sdata_idx, unsigned int tid, int threads)
{

  float myAggregate = sdata[tid];
  volatile float* smem = sdata;
  volatile float* smem_idx = sdata_idx;

  // do reduction in shared mem
  if (threads >= 512) { if (tid < 256) { myAggregate = reduce_value<strategy,0>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid, 256); } __syncthreads(); }
  if (threads >= 256) { if (tid < 128) { myAggregate = reduce_value<strategy,0>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid, 128); } __syncthreads(); }
  if (threads >= 128) { if (tid <  64) { myAggregate = reduce_value<strategy,0>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,  64); } __syncthreads(); }

  if (threads == 32){
    if (tid < 16)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      if (threads >=  32) { myAggregate = reduce_value<strategy,1>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,  16); }
      if (threads >=  16) { myAggregate = reduce_value<strategy,1>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,   8); }
      if (threads >=   8) { myAggregate = reduce_value<strategy,1>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,   4); }
      if (threads >=   4) { myAggregate = reduce_value<strategy,1>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,   2); }
      if (threads >=   2) { myAggregate = reduce_value<strategy,1>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,   1); }
    }
  }
  else
  {
    if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (threads >=  64) { myAggregate = reduce_value<strategy,1>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,  32); }
      if (threads >=  32) { myAggregate = reduce_value<strategy,1>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,  16); }
      if (threads >=  16) { myAggregate = reduce_value<strategy,1>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,   8); }
      if (threads >=   8) { myAggregate = reduce_value<strategy,1>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,   4); }
      if (threads >=   4) { myAggregate = reduce_value<strategy,1>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,   2); }
      if (threads >=   2) { myAggregate = reduce_value<strategy,1>(myAggregate, sdata, smem, sdata_idx, smem_idx, tid,   1); }
    }
  }
}




template <int strategy>
__device__ void kReduceRow(float *A, float *out_values, float *out_idxes, unsigned int rows, unsigned int cols)
{
	__shared__ float row_values[256];
	__shared__ float row_idxes[256];
	unsigned int idx = 0;
	float row_value = 0.0f;
	float row_idx = 0.0f;
	unsigned int next_thread_block_range = blockDim.x*2;

	if(strategy == row_sum || strategy == row_mean) row_values[threadIdx.x] = 0.0f;
	else row_values[threadIdx.x] = -FLT_MAX;
	__syncthreads();


	for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x)
	{
		if(strategy == row_sum || strategy == row_mean) row_value = 0.0f;
		else row_value = -FLT_MAX;
		for(unsigned int col = threadIdx.x; col < cols; col +=blockDim.x,next_thread_block_range+=blockDim.x)
		{
			idx = (col * rows) + row;
			row_values[threadIdx.x] = A[idx];
			__syncthreads();
			switch(strategy)//the compiler will optimize out this switch statement
			{
				case row_mean:
				case row_sum:
					reduce<row_sum>(row_values, row_idxes, threadIdx.x, blockDim.x);
					__syncthreads();
					if(threadIdx.x == 0) row_value += row_values[0];
					break;
				case row_max:
					reduce<row_max>(row_values, row_idxes, threadIdx.x, blockDim.x);
					__syncthreads();
					if(threadIdx.x == 0) row_value = fmaxf(row_value, row_values[0]);
					break;
				case row_argmax:
				case row_max_and_argmax:
					row_idxes[threadIdx.x] = col; __syncthreads();
					reduce<row_argmax>(row_values, row_idxes, threadIdx.x, blockDim.x);
					__syncthreads();
					if(threadIdx.x == 0)
					{
						if(row_value <= row_values[0])
						{
							row_value = row_values[0];
							row_idx = row_idxes[0];
						}
					}
					break;

			}

			if(cols< next_thread_block_range )
			{
				//the next block will be partially out of range
				//so that the values do not longer overlap (matrix and shared mem)
				//so we have to set the non-overlapping values to the default value
				if(strategy == row_sum || strategy == row_mean) row_values[threadIdx.x] = 0.0f;
				else row_values[threadIdx.x] = -FLT_MAX;
				__syncthreads();

			}
		}

		if(threadIdx.x == 0)
		{
			switch(strategy)
			{
				case row_mean: out_values[row] = row_value/((float)cols); break;
				case row_sum: out_values[row] = row_value; break;
				case row_max:out_values[row] = row_value; break;
				case row_argmax: out_values[row] = row_idx; break;
				case row_max_and_argmax: out_values[row] = row_value; out_idxes[row] = row_idx; break;

			}
		}

	}
}

__global__ void kReduceRow(float *A, float *out_values, float *out_idxes, unsigned int rows, unsigned int cols, RowReduction_t strategy)
{
	switch(strategy)//this is slow but more readable and maintainable
	{
		case row_mean: kReduceRow<row_mean>(A, out_values, out_idxes, rows, cols); break;
		case row_sum: kReduceRow<row_sum>(A, out_values, out_idxes, rows, cols); break;
		case row_max: kReduceRow<row_max>(A, out_values, out_idxes, rows, cols); break;
		case row_argmax: kReduceRow<row_argmax>(A, out_values, out_idxes, rows, cols); break;
		case row_max_and_argmax: kReduceRow<row_max_and_argmax>(A, out_values, out_idxes, rows, cols); break;

	}
}


__global__ void kMaxout(float *A, float *out, float *outargmax, int maxout_level, unsigned int cols, unsigned int rows)
{
  __shared__ float max_values[32];
  __shared__ float argmax_values[32];
  float const min_value = -FLT_MAX;

  for(int row = blockIdx.x; row < rows; row +=blockDim.x)
  {
	  int softout_block_idx = row + (blockIdx.y*maxout_level*rows);
	  if(threadIdx.x < maxout_level)
	  {
		  max_values[threadIdx.x] = A[softout_block_idx+(threadIdx.x*rows)];
		  argmax_values[threadIdx.x] = (float)((blockIdx.y*maxout_level)+threadIdx.x);
	  }
	  else
	  {
		  max_values[threadIdx.x] = min_value;
		  argmax_values[threadIdx.x] = -1.0f;
	  }

	  //reduceToMax(max_values, threadIdx.x);
	  reduceToMaxAndArgMax(max_values, argmax_values, threadIdx.x, 32);
	  __syncthreads();
	  if(threadIdx.x == 0) out[row + (blockIdx.y*rows)] = max_values[0];
	  if(threadIdx.x == 1) outargmax[row + (blockIdx.y*rows)] = argmax_values[0];
  }
}

__global__ void kSoftMax(float* A, float* out, unsigned int rows, unsigned int cols)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int mapOffset = rows*cols*blockIdx.y;
	unsigned int batchOffset = rows*cols*gridDim.y*blockIdx.x;
	float col_value = 0.0f;

	__shared__ float max_values[THREADS_PER_BLOCKS];
	__shared__ float row_sums[THREADS_PER_BLOCKS];

	for (unsigned int row = idx; row < rows; row += numThreads)
	{
		//fill with min values
		max_values[idx] = -FLT_MAX;
		row_sums[idx] = 0.0f;

		 //calc max value of the row
		for (unsigned int i = 0; i < cols; i++)
		{
			col_value = A[(i*rows) + row+mapOffset + batchOffset];
			if(col_value > max_values[idx])
			{
				max_values[idx] = col_value;
			}
		}

		//calc the row sum
		for (unsigned int i = 0; i < cols; i++)
		{
			row_sums[idx] += __expf(A[(i*rows) + row] - max_values[idx]);
		}

		//calc the value of each element in the row
		for (unsigned int i = 0; i < cols; i++)
		{
			out[(i*rows) + row+mapOffset + batchOffset] = __expf(A[(i*rows) + row] - max_values[idx])/row_sums[idx];
		}
	}

}

__global__ void kArgmax(float* A, float* out, unsigned int rows, unsigned int cols)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int mapOffset = rows*cols*blockIdx.y;
	unsigned int batchOffset = rows*cols*gridDim.y*blockIdx.x;
	float max_value = -FLT_MAX;
	float max_i = 0;
	float col_value = 0.0f;

	for (unsigned int row = idx; row < rows; row += numThreads)
	{
	  for (unsigned int i = 0; i < cols; i++)
	  {
		  col_value = A[(i*rows) + row +mapOffset+batchOffset];
		  if(col_value > max_value)
		  {
			  max_value = col_value;
			  max_i = i;
		  }

	  }
	  out[row+batchOffset+mapOffset] = max_i;
	}
}

//for column major data
__global__ void kSlice(float *A, float *out, int b1, int b2, int m1, int m2, int r1, int r2, int c1, int c2,  int rows, int cols, int batches_slice, int maps_slice, int cols_slice, int rows_slice, int is_forward_slice)
{
	int mapOffset = rows*cols;
	int batchOffset = mapOffset*gridDim.y;
	int mapOffsetSlice = rows_slice*cols_slice;
	int batchOffsetSlice = mapOffsetSlice*maps_slice;
	int batchidx=0;
	int mapidx=0;
	int colidx=0;
	int batchidx_slice = 0;
	int mapidx_slice = 0;
	int colidx_slice = 0;

	for(int batch = blockIdx.x+b1; batch < b2; batch+=gridDim.x)
	{
		batchidx = batch*batchOffset;
		batchidx_slice = (batch - b1)*batchOffsetSlice;
		for(int map = blockIdx.y+m1; map < m2; map+=gridDim.y)
		{
			mapidx = (map*mapOffset)+batchidx;
			mapidx_slice = ((map-m1)*mapOffsetSlice) + batchidx_slice;
			for(int col = threadIdx.y+c1; col < c2; col+=blockDim.y)
			{
				colidx =  (col*rows)+mapidx;
				colidx_slice = ((col-c1)*rows_slice) + mapidx_slice;
				for(int row = threadIdx.x+r1; row < r2; row+=blockDim.x)
				{
					if(is_forward_slice == 1)
						out[colidx_slice + (row-r1)] = A[colidx + (row < 0 ? (rows + row) : row)];
					else
						out[colidx + (row < 0 ? (rows + row) : row)] = A[colidx_slice + (row-r1)];
				}

			}
		}
	}
}


/*
//for column major data
__global__ void kSlice(float *A, float *out, Slice *S,  int rows, int cols, int batches_slice, int maps_slice, int cols_slice, int rows_slice, int is_forward_slice)
{
	int mapOffset = rows*cols;
	int batchOffset = mapOffset*gridDim.y;
	int mapOffsetSlice = rows_slice*cols_slice;
	int batchOffsetSlice = mapOffsetSlice*maps_slice;
	int batchidx=0;
	int mapidx=0;
	int colidx=0;
	int batchidx_slice = 0;
	int mapidx_slice = 0;
	int colidx_slice = 0;

	for(int batch = blockIdx.x+S->batch_start; batch < S->batch_stop; batch+=gridDim.x)
	{
		batchidx = batch*batchOffset;
		batchidx_slice = (batch - S->batch_start)*batchOffsetSlice;
		for(int map = blockIdx.y+S->map_start; map < S->map_stop; map+=gridDim.y)
		{
			mapidx = (map*mapOffset)+batchidx;
			mapidx_slice = ((map-S->map_start)*mapOffsetSlice) + batchidx_slice;
			for(int col = threadIdx.y+S->col_start; col < S->col_stop; col+=blockDim.y)
			{
				colidx =  (col*rows)+mapidx;
				colidx_slice = ((col-S->col_start)*rows_slice) + mapidx_slice;
				for(int row = threadIdx.x+S->row_start; row < S->row_stop; row+=blockDim.x)
				{
					if(is_forward_slice == 1)
						out[colidx_slice + (row-S->row_start)] = A[colidx + (row < 0 ? (rows + row) : row)];
					else
						out[colidx + (row < 0 ? (rows + row) : row)] = A[colidx_slice + (row-S->row_start)];
				}

			}
		}
	}
}
*/

//for column major data
__global__ void kAddScaledMatrixVector(float *A, float *v, float weight, float *out, int rows, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  //offset = current_column * rows
  int offset = 0;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  offset = (i / rows); //note: int arithmetic
	  out[i] =  A[i] + (v[offset]*weight);
  }
}

__global__ void kCreate_t_matrix(float *labels, float *out, int rows, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  int label = 0;
	  int offset = 0;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  label = (int)(labels[i]);
		  //offset = (label*rows) gives the current column; i gives the current row
		  offset = (label*rows) + i;
		  out[offset] = 1.0f;
	  }

}

__global__ void kDoubleRectifiedLinear(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  float value = 0.0f;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  value = (A[i] > 0.0f) ? A[i] : 0.0f;
      out[i] = (value < 1.0f) ? value : 1.0f;
  }
}

__global__ void kLinear(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
      out[i] = A[i];

}

__global__ void kDoubleRectifiedLinear_Derivative(float *A, float *out, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  out[i] = (A[i] <= 0.0f) || (A[i] >=1.0f) ? 0.0f : 1.0f;
	  }

}

__global__ void kHardTanH(float *A, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  float value = 0.0f;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  value = (A[i] > 1.0f) ? A[i] : 1.0f;
      out[i] = (value < -1.0f) ? value : -1.0f;
  }
}

__global__ void kPairwise_ranking(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  float value = 0.0f;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  value = 1.0f - A[i] + B[i];
      out[i] = value < 0.0f ? 0.0f : value;
  }
}

__global__ void kPairwise_ranking_derivative(float *A, float *B, float *out, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
      out[i] = (1.0f - A[i] + B[i]) > 0.0f ? 1.0f : 0.0f;

}

__global__ void kHardTanH_Derivative(float *A, float *out, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	  for (unsigned int i = idx;i < size; i += numThreads)
		  out[i] = (A[i] < -1.0f) || (A[i] >1.0f) ? 0.0f : 1.0f;

}

__global__ void kSquaredError(float *A, float *t, float *out, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	  for (unsigned int i = idx;i < size; i += numThreads)
		  out[i] = powf(A[i] -t[i],2.0f);
}


__global__ void kArange(float *out, int start, int rows, int cols, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  int offset = 0;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  offset = (i % rows)*cols;

		  out[i] = (float)(offset + (i/rows) + start);
	  }
}

__global__ void kDropout(float *A, float *rdm, float dropout, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	  for (unsigned int i = idx;i < size; i += numThreads)
		  rdm[i] = rdm[i] > dropout ? A[i] : 0.0f;

}

__global__ void kDropout_cached(float *A, float *dropout, float *out, int current_idx, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	  int shifted_idx = 0;
	  int offset = 0;
	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  shifted_idx = i +current_idx;
		  offset = shifted_idx/10000;
		  out[i] = dropout[shifted_idx - (offset*10000)] == 1.0f ? A[i] : 0.0f;
	  }

}

__global__ void kWeightUpdate(float *RMS, float *grad, float RMS_multiplier, float learning_rate, int batch_size, int size, weightUpdate_t strategy)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;

	  switch(strategy)
	  {
		  case RMSProp:
			  for (unsigned int i = idx;i < size; i += numThreads)
			  {
				  grad_value = fdividef(grad[i],(float)batch_size);
				  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);

				  grad[i] = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));
				  RMS[i] = RMS_value;
			  }
			  break;
	  }

}


__global__ void kRMSprop(float *RMS, float *grad, float RMS_multiplier, float learning_rate, int batch_size, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = fdividef(grad[i],(float)batch_size);
		  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);

		  grad[i] = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));
		  RMS[i] = RMS_value;
	  }

}

__global__ void kRMSprop_with_momentum_update (float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;
	  float momentum_matrix_value = 0.0f;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = fdividef(grad[i],(float)batch_size);
		  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
		  grad_value = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));
		  momentum_matrix_value = m[i];
		  momentum_matrix_value -= grad_value;

		  RMS[i] = RMS_value;
		  m[i] = momentum_matrix_value;
	  }
}

__global__ void kRMSprop_with_momentum_weight_update (float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;
	  float momentum_matrix_value = 0.0f;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = fdividef(grad[i],(float)batch_size);
		  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
		  grad_value = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));
		  momentum_matrix_value = m[i] = (momentum*momentum_matrix_value) - grad_value;

		  RMS[i] = RMS_value;
		  w[i] += momentum_matrix_value;

	  }
}

__global__ void kRMSprop_with_nesterov_weight_update (float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {

		  grad_value = fdividef(grad[i],(float)batch_size);
		  m[i] = (momentum*m[i]) - (learning_rate*grad_value);

		  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
		  grad_value = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));

		  RMS[i] = RMS_value;
		  w[i] -= grad_value;

		  /*
		  grad_value = learning_rate*fdividef(grad[i],(float)batch_size);
		  m[i] = (momentum*m[i]) - grad_value;
		  w[i] -= grad_value;
			*/
	  }
}

__global__ void kNesterov_weight_update (float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = learning_rate*fdividef(grad[i],(float)batch_size);
		  m[i] = (momentum*m[i]) - grad_value;
		  w[i] -= grad_value;

	  }
}


__global__ void kCompression_8bit_test(float *tbl, float *A, float precision, int size, float *out)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float absnumber = 0.0;
	float multiplier = 0.1f/precision;
	float threshold = precision/1.e6f;

	__shared__ float tbl_values[128];
	if(threadIdx.x < 126)
		tbl_values[threadIdx.x] = tbl[threadIdx.x];

	__syncthreads();

	  for (int i = idx;i < size; i += numThreads)
	  {
		  int isNegative = 0;
		  int pivot = 63;
		  int upper_pivot = 125;
		  int lower_pivot = 0;
		  absnumber = A[i]*multiplier;
		  if(absnumber < 0.0f){isNegative = 1; absnumber=-absnumber; }
		  if(absnumber < threshold){ out[i] = 0.0f; continue; }
		  for(int j = 32; j > 0; j>>=1)
		  {
			  if(absnumber > tbl_values[pivot])
			  {
				  lower_pivot = pivot;
				  pivot+=j;
			  }
			  else
			  {
				  upper_pivot = pivot;
				  pivot-=j;
			  }

		  }

		  if(lower_pivot == pivot)
			  if(fabsf(tbl_values[pivot]-absnumber) < (tbl_values[upper_pivot]-absnumber))
				  out[i] = tbl_values[pivot]/(isNegative == 1 ? -multiplier : multiplier);
			  else
				  out[i] = tbl_values[upper_pivot]/(isNegative == 1 ? -multiplier : multiplier);
		  else
			  if((tbl_values[pivot]-absnumber) < fabsf(tbl_values[lower_pivot]-absnumber))
				  out[i] = tbl_values[pivot]/(isNegative == 1 ? -multiplier : multiplier);
			  else
				  out[i] = tbl_values[lower_pivot]/(isNegative == 1 ? -multiplier : multiplier);



	  }
}

__global__ void kDecompression_8bit(float *flt_tbl, unsigned char *A, float precision, int size, float *out)
{

	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	__shared__ float tbl_floats[256];
	if(threadIdx.x < 126)
	{
		tbl_floats[threadIdx.x] = flt_tbl[threadIdx.x]*precision;
		tbl_floats[threadIdx.x+128] = -tbl_floats[threadIdx.x];
	}


	tbl_floats[126] = 0.0f;
	tbl_floats[254] = -0.0f;
	tbl_floats[127] = precision;
	tbl_floats[255] = -precision;

	__syncthreads();

	for (int i = idx;i < size; i += numThreads)
	{
		out[i] = tbl_floats[A[i]];
	}
}


__global__ void kCompression_8bit(float *flt_tbl, float *A, float precision, int size, unsigned char *out)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float absnumber = 0.0f;
	float threshold_lower = 0.0000015;
	float threshold_upper = 0.995703;
	int isNegative = 0;
	int pivot = 63;
	int upper_pivot = 125;
	int lower_pivot = 0;

	__shared__ float tbl_floats[128];
	if(threadIdx.x < 126)
		tbl_floats[threadIdx.x] = flt_tbl[threadIdx.x];


	__syncthreads();

	  for (int i = idx;i < size; i += numThreads)
	  {
		  isNegative = 0;
		  pivot = 63;
		  upper_pivot = 125;
		  lower_pivot = 0;
		  absnumber = A[i]/precision;
		  if(absnumber < 0.0f){isNegative = 1; absnumber=-absnumber; }
		  if(absnumber < threshold_lower){ out[i] = (unsigned char)126; continue; }
		  if(absnumber > threshold_upper){ out[i] = (isNegative == 0 ? (unsigned char)127 : (unsigned char)255); continue; }
		  for(int j = 32; j > 0; j>>=1)
		  {
			  if(absnumber > tbl_floats[pivot])
			  {
				  lower_pivot = pivot;
				  pivot+=j;
			  }
			  else
			  {
				  upper_pivot = pivot;
				  pivot-=j;
			  }

		  }

		  if(lower_pivot == pivot)
			  if(fabsf(tbl_floats[pivot]-absnumber) < (tbl_floats[upper_pivot]-absnumber))
				  if(isNegative == 1)
					  out[i] =  pivot | 1 << 7;
				  else
					  out[i] =  pivot;
			  else
				  if(isNegative == 1)
					  out[i] =  upper_pivot | 1 << 7;
				  else
					  out[i] =  upper_pivot;
		  else
			  if((tbl_floats[pivot]-absnumber) < fabsf(tbl_floats[lower_pivot]-absnumber))
				  if(isNegative == 1)
					  out[i] =  (pivot | 1 << 7);
				  else
					  out[i] =  pivot;
			  else
		  	  	  if(isNegative == 1)
		  	  		  out[i] =  lower_pivot | 1 << 7;
		  		  else
		  			  out[i] =  lower_pivot;

	  }
}


__global__ void kRMSprop_with_weight_update (float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  float grad_value = 0.0f;
	  float RMS_value = 0.0f;
	  float rms_reciprocal = 1.0f - RMS_multiplier;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = fdividef(grad[i],(float)batch_size) ;
		  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
		  grad_value = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));

		  RMS[i] = RMS_value;
		  w[i] -= grad_value;

	  }
}



__global__ void kRMSprop_with_weight_update_8bit(float *RMS, float *grad, float *w, float *m, float RMS_multiplier, float learning_rate, int batch_size, int size, float momentum)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	float grad_value = 0.0f;
	float RMS_value = 0.0f;
	float rms_reciprocal = 1.0f - RMS_multiplier;

	for (unsigned int i = idx;i < size; i += numThreads)
	{
	  grad_value = fdividef(grad[i],(float)batch_size);
	  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
	  grad[i] = learning_rate*fdividef(grad_value,(sqrtf(RMS_value)+1.0e-08f));

	  RMS[i] = RMS_value;

	}
}

__global__ void kSparseDot(int m, int n, int k, float *data, int* indptr, int* indices, float *dense_data, float* target, float beta, float alpha)
{
  const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n)
  {
	  /*
	  for(int i = 0; i < indptr[m+1];i++)
		  if(indices[i] > 23)
		  {
			  printf("ERROR: \n");
			  printf("%i \n", indices[i]);
	    	  printf("col: %i \n", col);
	    	  printf("row: %i \n", row);
		  }
		  */

	  int max_idx = indptr[m+1];
	  for(int i = 0; i < m+1;i++)
		  if(indptr[i] > max_idx)
		  {
			  printf("ERROR: \n");
			  printf("%i \n", indptr[i]);
	    	  printf("max_idx: %i \n", max_idx);
		  }


    const int start = indptr[row];
    const int end = indptr[row + 1];
    float sum = 0.f;
    for (int i = start; i < end; i++)
    {
    	/*
    	for(int a = start; a < end;a++)
    			  if(indices[a] > 23)
    			  {
    				  printf("ERROR: \n");
    				  printf("%i \n", indices[a]);
    		    	  printf("a: %i \n", a);
    			  }
    			  */


      sum += data[i]  * dense_data[(col * k) + indices[i]];
      if(sum > 500000 || sum < -500000)
      {

    	  printf("start: %i ", start);
    	  printf("end: %i ", end);
    	  printf("i: %i ", i);
    	  printf("k: %i ", k);
    	  printf("col: %i ", col);
    	  printf("data idx %i ", indices[i]);
    	  printf("full idx %i ", (col * k) + indices[i]);
    	  printf("data sparse %f ", data[i]);
    	  printf("data dense %f ", dense_data[col * k + indices[i]]);
    	 printf("data point %f ", data[i]  * dense_data[col * k + indices[i]]);
         printf(" sum %f\n", sum);



         return;
      }
    }
    const int pos = col * m + row;
    target[pos] = alpha * sum + ((beta == 0) ? 0 : beta * target[pos]);
  }
}

__global__ void kPrintData(float *A, int size)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	__syncthreads();
	if(idx == 0)
		printf("[");
	for (unsigned int i = idx;i < size; i += numThreads)
		 printf("%f ",A[i]);
	__syncthreads();
	if(idx == 0)
	printf("]\n");
}


__global__ void kMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height)
{
  extern __shared__ float max_vals[];
  float cur_max = -FLT_MAX;
  float val = 0;
  const int column = gridDim.x * blockIdx.y + blockIdx.x;
  if (column < width) {
    float *cur_data = &mat[column * height] ;
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
      val = cur_data[i];
      if (val > cur_max) cur_max = val;
    }
    max_vals[threadIdx.x] = cur_max;
    reduceToMax(max_vals, threadIdx.x, blockDim.x);
    __syncthreads();
    if (threadIdx.x == 0) target[column] = max_vals[0];
  }
}



__global__ void kExpandToMaxoutGrad(float* error, float* indexes, float *out, int error_size, int error_rows, int maxout_level)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int grad_size = maxout_level*error_size;

    for (unsigned int i = idx;i < grad_size; i += numThreads)
    	out[i] = 0.0f;

	for (unsigned int i = idx;i < error_size; i += numThreads)
	{
		int row_idx = idx - ((idx / error_rows)*error_rows);
		out[row_idx + (((int)indexes[idx])*error_rows)] = error[i];
	}
}

__global__ void kConstructVocabMatrix(float *vocab_idx, float *vocab_idx_y, float* vocab, float *rdm_idx, float *batch_X, float *batch_Y)
{
	int middleIdx = (gridDim.y/2);
	int myIdx = 0;
	int myRdmIdx = 0;

	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y

	//middle index is replaced by rdm word for batch_Y, but we still need to write the correct word into batch_X!
	if(blockIdx.y != middleIdx)
	{
		myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
		vocab_idx_y[blockIdx.x+(blockIdx.y*gridDim.x)] = (float)myIdx;
	}
	else
	{
		myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
		myRdmIdx = (int)rdm_idx[blockIdx.x];
		vocab_idx_y[blockIdx.x+(blockIdx.y*gridDim.x)] = (float)myRdmIdx;
	}

	int myVocabIdx = blockDim.x*myIdx;
	int myVocabRdmIdx = blockDim.x*myRdmIdx;

	if(blockIdx.y != middleIdx)
	{
		batch_X[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)] = vocab[myVocabIdx + threadIdx.x];
		batch_Y[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)] = vocab[myVocabIdx + threadIdx.x];
	}
	else
	{
		batch_X[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)] = vocab[myVocabIdx + threadIdx.x];
		batch_Y[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)] = vocab[myVocabRdmIdx + threadIdx.x];
	}



}


__global__ void concat_batches(float **batch_X, float **batch_Y, float *out_X, float *out_Y)
{
	//gridDim.z = matrix_count
	//gridDim.y = batch size
	//gridDim.x = window_size
	//blockDim.x = partial vocab size

	int full_vocab_size = gridDim.z*blockDim.x;
	int cols = gridDim.x*full_vocab_size;
	int partial_cols = blockDim.x*gridDim.x;

	//full_size times current row = current row idx
	//current window position times partial_threads times current matrix = current word idx
	//threadIdx.x current parameter within a word
	out_X[(blockIdx.y *cols) + (blockIdx.x*full_vocab_size) + (blockIdx.z*blockDim.x)  +threadIdx.x] = batch_X[blockIdx.z][(blockIdx.y *partial_cols) + (blockIdx.x*blockDim.x)  + threadIdx.x];
	out_Y[(blockIdx.y *cols) + (blockIdx.x*full_vocab_size) + (blockIdx.z*blockDim.x)  +threadIdx.x] = batch_Y[blockIdx.z][(blockIdx.y *partial_cols) + (blockIdx.x*blockDim.x)  + threadIdx.x];

}


/*
 //numerically unstable?
__global__ void kUpdateVocabWithGradient(float *grad, float *vocab_idx, float* vocab, float learning_rate)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y

	int myIdx = 0;
	float multiplier = -fdividef(learning_rate,float(gridDim.x));
	myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
	int myVocabIdx = blockDim.x*myIdx;



	//printf("%f ",grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]*multiplier);
	//printf("%f ",vocab[myVocabIdx + threadIdx.x]);
	//printf("%f ",vocab[myVocabIdx + threadIdx.x]+ (grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]*multiplier));
	if(myIdx > 10000)
		atomicAdd(&vocab[myVocabIdx + threadIdx.x],grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]*multiplier);
	//vocab[myVocabIdx + threadIdx.x] +=grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)];
	//printf("%s ",!isfinite(grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]*multiplier));

}
*/



//numerically unstable?
__global__ void kUpdateVocabWithGradient(float *grad, float *vocab_idx, float* vocab, float learning_rate)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y

	int myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
	int myVocabIdx = blockDim.x*myIdx;
	atomicAdd(&vocab[myVocabIdx + threadIdx.x],-grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]*learning_rate);
}






__global__ void kExpandDoubleVocabGradient(float *gradX, float *gradY, float *vocab_idx_X, float *vocab_idx_Y, float* vocab,
										 float *vocab_grad, float *vocab_grad_idx, float learning_rate, int grad_size)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y


	//float multiplier = fdividef(learning_rate,(float)(gridDim.x*2));
	int myIdx_X = (int)vocab_idx_X[blockIdx.x+(blockIdx.y*gridDim.x)];
	int myIdx_Y = (int)vocab_idx_Y[blockIdx.x+(blockIdx.y*gridDim.x)];
	//int grad_cols = grad_size/blockDim.x;

	int myVocabIdx_X = blockDim.x*myIdx_X;
	int myVocabIdx_Y = blockDim.x*myIdx_Y;


	atomicAdd(&vocab_grad[myVocabIdx_X + threadIdx.x],gradX[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]);
	atomicAdd(&vocab_grad[myVocabIdx_Y + threadIdx.x],gradY[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]);
	/*
	vocab_grad_idx[myIdx_X] = 1.0f;
	vocab_grad_idx[myIdx_Y] = 1.0f;

	__syncthreads();




	int block_idx = (blockIdx.y*gridDim.x) + blockIdx.x;
	int threads_blocks = gridDim.x*gridDim.y;
	for(int i = block_idx; i < grad_cols; i+=threads_blocks)
	{
		if(vocab_grad_idx[i] == 1.0f)
		{
			vocab[(i*blockDim.x) + threadIdx.x] -= vocab_grad[(i*blockDim.x) + threadIdx.x]*multiplier;
		}
	}

	*/


}


/*
__global__ void kExpandVocabGradient_sharedMemory(float *grad, float *vocab_idx, float *vocab_grad, float *sorted_vocab_idx, vocab_idx_size)
{
	//vocab_vector_size = blockDim.x;
	//batch_size = gridDim.x
	//try different configs for gridDim.x, e.g 16, 32 etc.

	//will have vocab_vector_size = blockDim.x elements e.g. 64
	extern __shared__ float sGrads[];

	float myWordIdx = 0.0f;
	float last_word = 0.0f;
	float currentIdx = 0.0f;

	sGrads[threadIdx.x] = 0.0f;

	for(int word = blockIdx.x; currentIdx < vocab_idx_size; word++)
	{
		for(int i = currentIdx; i < vocab_idx_size; i++, currentIdx++)
		{

		}
	}
}
*/


__global__ void kExpandVocabGradient(float *grad, float *vocab_idx, float *vocab_grad)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y

	int myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
	int myVocabIdx = blockDim.x*myIdx;
	atomicAdd(&vocab_grad[myVocabIdx + threadIdx.x],grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]);

}

__global__ void kExpandPartialVocabGradient(float *grad, float *vocab_idx, float *vocab_grad, int matrix_idx, int matrix_count)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y
	int offset = matrix_idx*gridDim.x*blockDim.x;
	int myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
	int myVocabIdx = blockDim.x*myIdx;
	atomicAdd(&vocab_grad[myVocabIdx + threadIdx.x],grad[blockIdx.x + (blockIdx.y*(blockDim.x*matrix_count)*gridDim.x) + (threadIdx.x*gridDim.x) + offset]);

}

__global__ void kExpandVocabGradientMiddleWord(float *grad, float *vocab_idx, float *vocab_grad)
{
	//vocab_vector_size = blockDim.x;
	//vocab_idx_rows = batch_size = gridDim.x
	//vocab_idx_cols = window_size = gridDim.y

	if(blockIdx.x+(blockIdx.y*gridDim.x) == gridDim.y/2)
	{
		int myIdx = (int)vocab_idx[blockIdx.x+(blockIdx.y*gridDim.x)];
		int myVocabIdx = blockDim.x*myIdx;
		atomicAdd(&vocab_grad[myVocabIdx + threadIdx.x],grad[blockIdx.x + (blockIdx.y*blockDim.x*gridDim.x) + (threadIdx.x*gridDim.x)]);
	}

}




__global__ void kDot8bit(unsigned char *A, unsigned char *B, float *out, int rowsA, int colsA, int colsB, float *flt_tbl, float precisionA, float precisionB)
{
	const unsigned int threads_per_block = blockDim.x*blockDim.y;
	const int mygrid = blockIdx.x;
	const int myidx = (threadIdx.y*blockDim.x)+threadIdx.x;

	__shared__ float tbl_floatsA[256];
	__shared__ float tbl_floatsB[256];
	for(int i = myidx; i < 126; i++)
	{
		tbl_floatsA[i] = flt_tbl[i]*precisionA;
		tbl_floatsA[i+128] = -tbl_floatsA[i];
		tbl_floatsB[i] = flt_tbl[i]*precisionB;
		tbl_floatsB[i+128] = -tbl_floatsB[i];
	}
	tbl_floatsA[126] = 0.0f;
	tbl_floatsB[126] = 0.0f;
	tbl_floatsA[127] = precisionA;
	tbl_floatsB[127] = -precisionA;
	tbl_floatsA[254] = -0.0f;
	tbl_floatsB[254] = -0.0f;
	tbl_floatsA[255] = precisionB;
	tbl_floatsB[255] = -precisionB;

	__syncthreads();



	for(int Arow = mygrid; Arow < rowsA; Arow+=gridDim.x)
	{
		for(int Bcol = myidx; Bcol < colsB; Bcol+=threads_per_block)
		{
			int idxout = (Bcol*rowsA) + Arow;
			for(int Acol = 0; Acol < colsA; Acol++)
				out[idxout] += tbl_floatsA[A[(Acol*rowsA)+Arow]] * tbl_floatsB[B[(colsA*Bcol)  + Acol]];

		}


	}





}

__global__ void kDot8bit_shared(unsigned char *A, unsigned char *B, float *out, int rowsA, int colsA, int colsB, float *flt_tbl, float precisionA, float precisionB)
{
	int myidx = (threadIdx.y*blockDim.x)+threadIdx.x;

	__shared__ unsigned char A_tile[64][256]; //64x32 banks
	__shared__ unsigned char B_tile[64][256];//256x8 banks

	__shared__ float tbl_floatsA[256];
	__shared__ float tbl_floatsB[256];
	for(int i = myidx; i < 126; i++)
	{
		tbl_floatsA[i] = flt_tbl[i]*precisionA;
		tbl_floatsA[i+128] = -tbl_floatsA[i];
		tbl_floatsB[i] = flt_tbl[i]*precisionB;
		tbl_floatsB[i+128] = -tbl_floatsB[i];
	}
	tbl_floatsA[126] = 0.0f;
	tbl_floatsB[126] = 0.0f;
	tbl_floatsA[127] = precisionA;
	tbl_floatsB[127] = -precisionA;
	tbl_floatsA[254] = -0.0f;
	tbl_floatsB[254] = -0.0f;
	tbl_floatsA[255] = precisionB;
	tbl_floatsB[255] = -precisionB;

	__syncthreads();


	int offset = 0;
	myidx = threadIdx.y*16;
	int Arow = threadIdx.x+(blockIdx.x*64);
	int Acol = (threadIdx.y*16)+(blockIdx.y*256);



	if(Arow < rowsA)
	{
		for(int i = 0; i < 16; i++){ A_tile[threadIdx.x][myidx+i] = A[((Acol+i)*rowsA)+ Arow]; }
		for(int i = threadIdx.y; i < colsB; i+=blockDim.y){ out[((i)*rowsA) + Arow] = 0.0f; }
	}
	else
		for(int i = 0; i < 16; i++){ A_tile[threadIdx.x][myidx+i] = 126; }

	for(int Btile = 0 ; Btile < colsB; Btile+=64)
	{
		if(Btile+threadIdx.x  < colsB)
		{
			for(int i = 0; i < 16; i++)
			{
				if(Acol+i < colsA)
					B_tile[threadIdx.x][myidx+i] = B[((threadIdx.x + Btile)*colsA)+ Acol+i];//B_tile is transposed to avoid bank conflicts with 64 threads
				else
					B_tile[threadIdx.x][myidx+i] = 126;
			}
		}
		else
		{
			for(int i = 0; i < 16; i++)
				B_tile[threadIdx.x][myidx+i] = 126;//B_tile is transposed to avoid bank conflicts with 64 threads
		}

		__syncthreads();
			for(int Bcol2 = offset; Bcol2 < 64 + offset; Bcol2++)
			{
					for (int i = 0; i < 16; ++i)
						atomicAdd(&out[((Bcol2)*rowsA) + Arow],
								tbl_floatsA[A_tile[threadIdx.x][myidx + i]] *
								tbl_floatsB[B_tile[Bcol2-offset][myidx + i]]);

			}

		offset +=64;
	}
}


__global__ void kCompression_1bit(float *A_with_errors, float *error,  float *avgPos, float *avgNeg, unsigned int *out_quant, int rows, int cols)
{
	float pos = 0.0f;
	float neg = 0.0f;
	unsigned int idx = 0;
	unsigned int bit = 0;
	unsigned int quant = 0.0;
	float value = 0.0f;
	for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x)
	{
		pos = avgPos[row];
		neg = avgNeg[row];
		for(unsigned int col = threadIdx.x; col < cols; col +=blockDim.x)
		{
			idx = (col * rows) + row;
			value = A_with_errors[idx];
			if(value >= 0)
			{
				bit = 1;
				error[idx] = value-pos;
			}
			else
			{
				bit = 0;
				error[idx] = value-neg;
			}

			quant = __ballot(bit);

			out_quant[(rows * (col/32)) + row] = quant;
		}
	}

}

__global__ void kDecompression_1bit(unsigned int *A_quant, float *error,  float *avgPos, float *avgNeg, float *out,  int rows, int cols)
{
	float pos = 0.0f;
	float neg = 0.0f;
	unsigned int idx = 0;
	unsigned int bit = 0;
	unsigned quant = 0.0;
	for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x)
	{
		pos = avgPos[row];
		neg = avgNeg[row];
		for(unsigned int col = threadIdx.x; col < cols; col +=blockDim.x)
		{
			idx = (rows * (col/32) + row);
			quant = A_quant[idx];
			bit = quant & (1<<(threadIdx.x % 32));
			out[(rows * col) + row] = bit ? pos : neg;
		}
	}

}


__global__ void kCompression_16bit(float *A, unsigned short *out, int size)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

		  for (unsigned int i = idx;i < size; i += numThreads)
			  out[i] = __float2half_rn(A[i]);
}

__global__ void kDecompression_16bit(unsigned short *A, float *out, int size)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

		  for (unsigned int i = idx;i < size; i += numThreads)
			  out[i] = __half2float(A[i]);
}


