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
	Slice *femptySlice(){ return emptySlice(); }
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

	Tensor *fadd(Tensor *A, Tensor *B){ return applyFunc(A,B,add_tensor); }
	void inp_add(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,add_tensor); }
	Tensor *fsub(Tensor *A, Tensor *B){ return applyFunc(A,B,sub_tensor); }
	void inp_sub(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,sub_tensor); }
	Tensor *fmul(Tensor *A, Tensor *B){ return applyFunc(A,B,mul_tensor); }
	void inp_mul(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,mul_tensor); }
	Tensor *fdiv(Tensor *A, Tensor *B){ return applyFunc(A,B,div_tensor); }
	void inp_div(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,div_tensor); }
	void ffree(Tensor *A){ A->freeTensor(); }

	Tensor *fscalarAdd(Tensor *A, float a){ return applyFunc(A,NULL,a,add_scalar); }
	void inp_scalarAdd(Tensor *A, float a, Tensor *out){ applyFunc(A,NULL,out,a,add_scalar); }
	Tensor *fscalarSub(Tensor *A, float a){ return applyFunc(A,NULL,-a,add_scalar); }
	void inp_scalarSub(Tensor *A, float a, Tensor *out){ applyFunc(A,NULL,out,-a,add_scalar); }
	Tensor *fscalarMul(Tensor *A, float a){ return applyFunc(A,NULL,a,mul_scalar); }
	void inp_scalarMul(Tensor *A, float a, Tensor *out){ applyFunc(A,NULL,out,a,mul_scalar); }
	Tensor *fscalarDiv(Tensor *A, float a){ return applyFunc(A,NULL,1.0f/a,mul_scalar); }
	void inp_scalarDiv(Tensor *A, float a, Tensor *out){ applyFunc(A,NULL,out,1.0f/a,mul_scalar); }


	Tensor *fexp(Tensor *A){ return applyFunc(A,NULL,exp_tensor); }
	void inp_exp(Tensor *A, Tensor *out){ applyFunc(A,NULL,out,exp_tensor);}
	Tensor *flog(Tensor *A){ return applyFunc(A,NULL,log_tensor);}
	void inp_log(Tensor *A, Tensor *out){ applyFunc(A,NULL,out,log_tensor);}
	Tensor *flogistic(Tensor *A){ return applyFunc(A,NULL,logistic);}
	void inp_logistic(Tensor *A, Tensor *out){ applyFunc(A,NULL,out, logistic);}
	Tensor *flogisticGrad(Tensor *A){ return applyFunc(A, NULL, logistic_grad); }
	void inp_logisticGrad(Tensor *A, Tensor *out){ applyFunc(A,NULL,out, logistic_grad); }
	Tensor *ffabs(Tensor *A){ return applyFunc(A, NULL, abs_tensor); }
	void inp_abs(Tensor *A, Tensor *out){ applyFunc(A,NULL,out,abs_tensor); }
	Tensor *ffpow(Tensor *A, float power){ return applyFunc(A,NULL,power,pow_tensor); }
	void inp_pow(Tensor *A, float power, Tensor *out){ applyFunc(A,NULL,out,power,pow_tensor); }

	Tensor *faddVectorToTensor(Tensor *A, Tensor *v){ return applyFunc(A,v,add_vec); }
	void inp_addVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,add_vec); }
	Tensor *fsubVectorToTensor(Tensor *A, Tensor *v){ return applyFunc(A,v,sub_vec); }
	void inp_subVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,sub_vec); }
	Tensor *fmulVectorToTensor(Tensor *A, Tensor *v){ return applyFunc(A,v,mul_vec); }
	void inp_mulVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,mul_vec); }
	Tensor *fdivVectorToTensor(Tensor *A, Tensor *v){ return applyFunc(A,v,div_vec); }
	void inp_divVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,div_vec); }

	Tensor *feq(Tensor *A, Tensor *B){ return applyFunc(A,B,eq_tensor); }
	void inp_eq(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,eq_tensor); }
	Tensor *flt(Tensor *A, Tensor *B){ return applyFunc(A,B,lt_tensor); }
	void inp_lt(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,lt_tensor); }
	Tensor *fgt(Tensor *A, Tensor *B){ return applyFunc(A,B,gt_tensor); }
	void inp_gt(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,gt_tensor); }
	Tensor *fge(Tensor *A, Tensor *B){ return applyFunc(A,B,ge_tensor); }
	void inp_ge(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,ge_tensor); }
	Tensor *fle(Tensor *A, Tensor *B){ return applyFunc(A,B,le_tensor); }
	void inp_le(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,le_tensor); }
	Tensor *fne(Tensor *A, Tensor *B){ return applyFunc(A,B,ne_tensor); }
	void inp_ne(Tensor *A, Tensor *B, Tensor *out){ applyFunc(A,B,out,ne_tensor); }

	Tensor *fvec_eq(Tensor *A, Tensor *v){ return applyFunc(A,v,eq_vec); }
	void inp_vec_eq(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,eq_vec); }
	Tensor *fvec_lt(Tensor *A, Tensor *v){ return applyFunc(A,v,lt_vec); }
	void inp_vec_lt(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,lt_vec); }
	Tensor *fvec_gt(Tensor *A, Tensor *v){ return applyFunc(A,v,gt_vec); }
	void inp_vec_gt(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,gt_vec); }
	Tensor *fvec_le(Tensor *A, Tensor *v){ return applyFunc(A,v,le_vec); }
	void inp_vec_le(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,le_vec); }
	Tensor *fvec_ge(Tensor *A, Tensor *v){ return applyFunc(A,v,ge_vec); }
	void inp_vec_ge(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,ge_vec); }
	Tensor *fvec_ne(Tensor *A, Tensor *v){ return applyFunc(A,v,ne_vec); }
	void inp_vec_ne(Tensor *A, Tensor *v, Tensor *out){ applyFunc(A,v,out,ne_vec); }

	Tensor *fscalar_eq(Tensor *A, float flt){ return applyFunc(A,NULL,flt,eq_scalar); }
	void inp_scalar_eq(Tensor *A, float flt, Tensor *out){ applyFunc(A,NULL,out,flt,eq_scalar); }
	Tensor *fscalar_lt(Tensor *A, float flt){ return applyFunc(A,NULL,flt,lt_scalar); }
	void inp_scalar_lt(Tensor *A, float flt, Tensor *out){ applyFunc(A,NULL,out,flt,lt_scalar); }
	Tensor *fscalar_gt(Tensor *A, float flt){ return applyFunc(A,NULL,flt,gt_scalar); }
	void inp_scalar_gt(Tensor *A, float flt, Tensor *out){ applyFunc(A,NULL,out,flt,gt_scalar); }
	Tensor *fscalar_le(Tensor *A, float flt){ return applyFunc(A,NULL,flt,le_scalar); }
	void inp_scalar_le(Tensor *A, float flt, Tensor *out){ applyFunc(A,NULL,out,flt,le_scalar); }
	Tensor *fscalar_ge(Tensor *A, float flt){ return applyFunc(A,NULL,flt,ge_scalar); }
	void inp_scalar_ge(Tensor *A, float flt, Tensor *out){ applyFunc(A,NULL,out,flt,ge_scalar); }
	Tensor *fscalar_ne(Tensor *A, float flt){ return applyFunc(A,NULL,flt,ne_scalar); }
	void inp_scalar_ne(Tensor *A, float flt, Tensor *out){ applyFunc(A,NULL,out,flt,ne_scalar); }

	Tensor *fslice(Tensor *A, Slice *S){ return applySliceFunc(A,S); }

	Tensor *fdot(GPUpy *gpupy, Tensor *A, Tensor *B){ return gpupy->dot(A,B); }
	void inp_dot(GPUpy *gpupy, Tensor *A, Tensor *B, Tensor *out){ gpupy->dot(A,B, out); }
	Tensor *fTdot(GPUpy *gpupy, Tensor *A, Tensor *B){ return gpupy->Tdot(A,B); }
	void inp_Tdot(GPUpy *gpupy, Tensor *A, Tensor *B, Tensor *out){ gpupy->Tdot(A,B, out); }
	Tensor *fdotT(GPUpy *gpupy, Tensor *A, Tensor *B){ return gpupy->dotT(A,B); }
	void inp_dotT(GPUpy *gpupy, Tensor *A, Tensor *B, Tensor *out){ gpupy->dotT(A,B, out); }

}
