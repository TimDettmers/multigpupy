/*
 * Tensor.cuh
 *
 *  Created on: Mar 21, 2015
 *      Author: tim
 */

#ifndef TENSOR_CUH_
#define TENSOR_CUH_

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cudaKernels.cuh>
#include <iostream>
#include <vector>
#include <float.h>
#include <limits.h>
#include <assert.h>
//this looks strange, but otherwise we cannot use template classes together with export "C" for ctypes
#include <Tensor.cu>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


#define TILE_DIM (32)
#define BLOCK_ROWS (8)
#define COPY_BLOCK_SIZE 16

#define RDM_NUMBERS_PER_THREAD (1024)
#define THREADS_PER_BLOCKS (512)
#define BLOCKS (4096)

#define DOT_BLOCKS (128)
#define TILE_SIZE (32)
#define DOT_REPS (4)



typedef TensorTemplate<float> Tensor;
typedef TensorTemplate<unsigned char> CharTensor;
typedef TensorTemplate<unsigned int> UIntTensor;
typedef TensorTemplate<unsigned short> UShortTensor;




#endif /* TENSOR_CUH_ */

