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
	Tensor *fsub(Tensor *A, Tensor *B){ return sub(A,B); }
	void inp_sub(Tensor *A, Tensor *B, Tensor *out){ return sub(A,B,out); }
	Tensor *fmul(Tensor *A, Tensor *B){ return mul(A,B); }
	void inp_mul(Tensor *A, Tensor *B, Tensor *out){ return mul(A,B,out); }
	Tensor *fdiv(Tensor *A, Tensor *B){ return div(A,B); }
	void inp_div(Tensor *A, Tensor *B, Tensor *out){ return div(A,B,out); }
	void ffree(Tensor *A){ A->freeTensor(); }

	Tensor *fscalarAdd(Tensor *A, float a){ return scalarAdd(A,a); }
	void inp_scalarAdd(Tensor *A, float a, Tensor *out){ scalarAdd(A,a, out); }
	Tensor *fscalarMul(Tensor *A, float a){ return scalarMul(A,a); }
	void inp_scalarMul(Tensor *A, float a, Tensor *out){ scalarMul(A,a, out); }
}
