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

	Tensor *fadd(Tensor *A, Tensor *B){ return applyFunc(A,B,opAdd); }
	void inp_add(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,opAdd); }
	Tensor *fsub(Tensor *A, Tensor *B){ return applyFunc(A,B,opSub); }
	void inp_sub(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,opSub); }
	Tensor *fmul(Tensor *A, Tensor *B){ return applyFunc(A,B,opMul); }
	void inp_mul(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,opMul); }
	Tensor *fdiv(Tensor *A, Tensor *B){ return applyFunc(A,B,opDiv); }
	void inp_div(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,opDiv); }
	void ffree(Tensor *A){ A->freeTensor(); }

	Tensor *fscalarAdd(Tensor *A, float a){ return applyFunc(A,NULL,a,add_scalar); }
	void inp_scalarAdd(Tensor *A, float a, Tensor *out){ applyFunc(A,NULL,out,a,add_scalar); }
	Tensor *fscalarMul(Tensor *A, float a){ return applyFunc(A,NULL,a,mul_scalar); }
	void inp_scalarMul(Tensor *A, float a, Tensor *out){ applyFunc(A,NULL,out,a,mul_scalar); }


	Tensor *fexp(Tensor *A){ return applyFunc(A,NULL,exp_tensor); }
	void inp_exp(Tensor *A, Tensor *out){ applyFunc(A,NULL,out,exp_tensor);}
	Tensor *fsqrt(Tensor *A){ return applyFunc(A,NULL,sqrt_tensor); }
	void inp_sqrt(Tensor *A, Tensor *out){ applyFunc(A,NULL,out,sqrt_tensor); }
	Tensor *flog(Tensor *A){ return applyFunc(A,NULL,log_tensor);}
	void inp_log(Tensor *A, Tensor *out){ applyFunc(A,NULL,out,log_tensor);}
	Tensor *flogistic(Tensor *A){ return applyFunc(A,NULL,logistic);}
	void inp_logistic(Tensor *A, Tensor *out){ applyFunc(A,NULL,out, logistic);}
	Tensor *flogisticGrad(Tensor *A){ return applyFunc(A, NULL, logistic_grad); }
	void inp_logisticGrad(Tensor *A, Tensor *out){ applyFunc(A,NULL,out, logistic_grad); }
	Tensor *ffabs(Tensor *A){ return applyFunc(A, NULL, abs_tensor); }
	void inp_abs(Tensor *A, Tensor *out){ applyFunc(A,NULL,out,abs_tensor); }
	Tensor *fsquare(Tensor *A){ return applyFunc(A,NULL,2.0f,pow_tensor); }
	void inp_square(Tensor *A, Tensor *out){ applyFunc(A,NULL,out,2.0f,pow_tensor); }
	Tensor *ffpow(Tensor *A, float power){ return applyFunc(A,NULL,power,pow_tensor); }
	void inp_pow(Tensor *A, float power, Tensor *out){ applyFunc(A,NULL,out,power,pow_tensor); }

	Tensor *faddVectorToTensor(Tensor *A, Tensor *v){ return applyFunc(A,v,addvec); }
	void inp_addVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,addvec); }
	Tensor *fsubVectorToTensor(Tensor *A, Tensor *v){ return applyFunc(A,v,subvec); }
	void inp_subVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,subvec); }
	Tensor *fmulVectorToTensor(Tensor *A, Tensor *v){ return applyFunc(A,v,mulvec); }
	void inp_mulVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,mulvec); }
	Tensor *fdivVectorToTensor(Tensor *A, Tensor *v){ return applyFunc(A,v,divvec); }
	void inp_divVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,divvec); }

}
