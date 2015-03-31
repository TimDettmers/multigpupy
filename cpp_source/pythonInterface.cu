/*
 * export.c
 *
 *  Created on: Mar 21, 2015
 *      Author: tim
 */

#include <basics.cuh>
#include <gpupy.cuh>
#include <time.h>

extern "C"
{
	GPUpy *fGPUpy(){GPUpy *gpupy = new GPUpy(); gpupy->init((int) ((time(0) + (12345)) % 10000)); return gpupy; }
	GPUpy *fseeded_GPUpy(int seed){GPUpy *gpupy = new GPUpy(); gpupy->init((int) ((time(0) + (12345)) % 10000)); return gpupy; }
	Tensor *fempty(int batches, int maps, int rows, int cols){ return empty(batches, maps, rows, cols); }
	Tensor *fzeros(int batches, int maps, int rows, int cols){ return zeros(batches, maps, rows, cols); }
	Tensor *fones(int batches, int maps, int rows, int cols){ return ones(batches, maps, rows, cols); }
	Tensor *ftocpu(Tensor *A, float *cpu_buffer){ return tocpu(A,cpu_buffer); }
	Tensor *fT(Tensor *A){ return T(A); }
	void inp_T(Tensor *A, Tensor *out){ T(A, out, A->cols,A->rows); }
	void ftogpu(Tensor *out, float *cpu_buffer){ togpu(out,cpu_buffer); }
	Tensor *frand(GPUpy *gpupy, int batches, int maps, int rows, int cols){ return gpupy->rand(batches, maps, rows, cols);  }
	Tensor *frandn(GPUpy *gpupy, int batches, int maps, int rows, int cols){ return gpupy->randn(batches, maps, rows, cols);  }
	Tensor *fnormal(GPUpy *gpupy, int batches, int maps, int rows, int cols, float mean, float std){ return gpupy->normal(batches, maps, rows, cols, mean, std);  }

	Tensor *fadd(Tensor *A, Tensor *B){ return add(A,B); }
	void inp_add(Tensor *A, Tensor *B, Tensor *out){ return add(A,B,out); }
	void ffree(Tensor *A){ A->freeTensor(); }
}
