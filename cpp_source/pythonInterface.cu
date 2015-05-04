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
	void fallocateNextAsync(GPUpy *gpupy, Tensor *A, float *cpu_buffer,float *pinned_A, Tensor *B, float *cpu_buffer_y, float *pinned_B, int batch_start_idx, int isSplit)
	{ gpupy->allocateNextAsync(A,cpu_buffer,pinned_A,B,cpu_buffer_y,pinned_B,batch_start_idx,isSplit); }
	void freplaceCurrentBatch(GPUpy *gpupy, Tensor *X, Tensor *y, Tensor *buffer, Tensor *buffer_y){ gpupy->replaceCurrentBatch(X, y, buffer, buffer_y); }
	float *fto_pinned(int batches, int maps, int rows, int cols, float *cpu_buffer){ return empty_pinned(batches,maps,rows,cols,cpu_buffer); }

	//GPUpy *fGPUpy(float *floats_8bit){GPUpy *gpupy = new GPUpy(); gpupy->init((int) ((time(0) + (12345)) % 10000),floats_8bit); return gpupy; }
	GPUpy *fGPUpy(float *floats_8bit){GPUpy *gpupy = new GPUpy(); gpupy->init(12345,floats_8bit); return gpupy; }
	GPUpy *fseeded_GPUpy(int seed,float *floats_8bit){GPUpy *gpupy = new GPUpy(); gpupy->init((int) ((time(0) + (12345)) % 10000),floats_8bit); return gpupy; }
	Slice *femptySlice(){ return emptySlice(); }
	Tensor *fempty(int batches, int maps, int rows, int cols){ return empty(batches, maps, rows, cols); }
	Tensor *fempty_like(Tensor *A){ return empty_like(A); }
	CharTensor *fempty_char_like(Tensor *A){ return empty_char_like(A); }
	UIntTensor *fempty_uint_like(Tensor *A){ return empty_uint_like(A); }
	UShortTensor *fempty_ushort_like(Tensor *A){ return empty_ushort_like(A); }
	Tensor *fempty_split(int batches, int maps, int rows, int cols, int split_axis){ return empty(batches, maps, rows, cols, split_axis); }
	Tensor *fzeros(int batches, int maps, int rows, int cols){ return zeros(batches, maps, rows, cols); }
	Tensor *fzeros_split(int batches, int maps, int rows, int cols, int split_axis){ return zeros(batches, maps, rows, cols, split_axis); }
	Tensor *fones(int batches, int maps, int rows, int cols){ return ones(batches, maps, rows, cols); }
	Tensor *ftocpu(Tensor *A, float *cpu_buffer){ return tocpu(A,cpu_buffer); }
	Tensor *fT(Tensor *A){ return T(A); }
	void inp_T(Tensor *A, Tensor *out){ T(A, out, A->cols,A->rows); }
	void ftogpu(Tensor *out, float *cpu_buffer){ togpu(out,cpu_buffer); }
	void inp_to_col_major_pinned(float *A_data, float *out_data, int batches, int maps, int rows, int cols){ to_col_major_pinned(A_data, out_data, batches, maps, rows, cols); }
	Tensor *frand(GPUpy *gpupy, int batches, int maps, int rows, int cols){ return gpupy->rand(batches, maps, rows, cols);  }
	Tensor *frandn(GPUpy *gpupy, int batches, int maps, int rows, int cols){ return gpupy->randn(batches, maps, rows, cols);  }
	Tensor *fnormal(GPUpy *gpupy, int batches, int maps, int rows, int cols, float mean, float std){ return gpupy->normal(batches, maps, rows, cols, mean, std);  }

	Tensor *fadd(Tensor *A, Tensor *B){ return elementWise(A,B,add_tensor); }
	void inp_add(Tensor *A, Tensor *B, Tensor *out){ elementWise(A,B,out,add_tensor); }
	Tensor *fsub(Tensor *A, Tensor *B){ return elementWise(A,B,sub_tensor); }
	void inp_sub(Tensor *A, Tensor *B, Tensor *out){ elementWise(A,B,out,sub_tensor); }
	Tensor *fmul(Tensor *A, Tensor *B){ return elementWise(A,B,mul_tensor); }
	void inp_mul(Tensor *A, Tensor *B, Tensor *out){ elementWise(A,B,out,mul_tensor); }
	Tensor *fdiv(Tensor *A, Tensor *B){ return elementWise(A,B,div_tensor); }
	void inp_div(Tensor *A, Tensor *B, Tensor *out){ elementWise(A,B,out,div_tensor); }
	void ffree(Tensor *A){ A->freeTensor(); }

	void ffill(Tensor *A, float value){ elementWise(A, NULL, NULL, value, fill); }

	void ffprint(Tensor *A){ elementWise(A,NULL,NULL,0.0f,print); }

	Tensor *fcopy(Tensor *A){ return elementWise(A,NULL,0.0f,copy); }
	void inp_copy(Tensor *A, Tensor *out){ elementWise(A,NULL,out,0.0f, copy); }

	Tensor *fscalarAdd(Tensor *A, float a){ return elementWise(A,NULL,a,add_scalar); }
	void inp_scalarAdd(Tensor *A, float a, Tensor *out){ elementWise(A,NULL,out,a,add_scalar); }
	Tensor *fscalarSub(Tensor *A, float a){ return elementWise(A,NULL,-a,add_scalar); }
	void inp_scalarSub(Tensor *A, float a, Tensor *out){ elementWise(A,NULL,out,-a,add_scalar); }
	Tensor *fscalarMul(Tensor *A, float a){ return elementWise(A,NULL,a,mul_scalar); }
	void inp_scalarMul(Tensor *A, float a, Tensor *out){ elementWise(A,NULL,out,a,mul_scalar); }
	Tensor *fscalarDiv(Tensor *A, float a){ return elementWise(A,NULL,1.0f/a,mul_scalar); }
	void inp_scalarDiv(Tensor *A, float a, Tensor *out){ elementWise(A,NULL,out,1.0f/a,mul_scalar); }


	Tensor *fexp(Tensor *A){ return elementWise(A,NULL,exp_tensor); }
	void inp_exp(Tensor *A, Tensor *out){ elementWise(A,NULL,out,exp_tensor);}
	Tensor *flog(Tensor *A){ return elementWise(A,NULL,log_tensor);}
	void inp_log(Tensor *A, Tensor *out){ elementWise(A,NULL,out,log_tensor);}
	Tensor *ffabs(Tensor *A){ return elementWise(A, NULL, abs_tensor); }
	void inp_abs(Tensor *A, Tensor *out){ elementWise(A,NULL,out,abs_tensor); }
	Tensor *ffpow(Tensor *A, float power){ return elementWise(A,NULL,power,pow_tensor); }
	void inp_pow(Tensor *A, float power, Tensor *out){ elementWise(A,NULL,out,power,pow_tensor); }


	Tensor *flogistic(Tensor *A){ return elementWise(A,NULL,logistic);}
	void inp_logistic(Tensor *A, Tensor *out){ elementWise(A,NULL,out, logistic);}
	Tensor *flogistic_grad(Tensor *A){ return elementWise(A, NULL, logistic_grad); }
	void inp_logistic_grad(Tensor *A, Tensor *out){ elementWise(A,NULL,out, logistic_grad); }
	Tensor *fReLU(Tensor *A){ return elementWise(A,NULL,rectified_linear);}
	void inp_ReLU(Tensor *A, Tensor *out){ elementWise(A,NULL,out, rectified_linear);}
	Tensor *fdouble_ReLU(Tensor *A){ return elementWise(A,NULL,double_rectified_linear);}
	void inp_double_ReLU(Tensor *A, Tensor *out){ elementWise(A,NULL,out, double_rectified_linear);}
	Tensor *fdouble_ReLU_grad(Tensor *A){ return elementWise(A,NULL,double_rectified_linear_grad);}
	void inp_double_ReLU_grad(Tensor *A, Tensor *out){ elementWise(A,NULL,out, double_rectified_linear_grad);}

	Tensor *fsoftmax(Tensor *A){return softmax(A); }
	void inp_softmax(Tensor *A, Tensor *out){ softmax(A, out); }
	Tensor *fargmax(Tensor *A){return argmax(A); }
	void inp_argmax(Tensor *A, Tensor *out){ argmax(A, out); }

	Tensor *faddVectorToTensor(Tensor *A, Tensor *v){ return vectorWise(A,v,add_vec); }
	void inp_addVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ vectorWise(A,v,out,add_vec); }
	Tensor *fsubVectorToTensor(Tensor *A, Tensor *v){ return vectorWise(A,v,sub_vec); }
	void inp_subVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ vectorWise(A,v,out,sub_vec); }
	Tensor *fmulVectorToTensor(Tensor *A, Tensor *v){ return vectorWise(A,v,mul_vec); }
	void inp_mulVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ vectorWise(A,v,out,mul_vec); }
	Tensor *fdivVectorToTensor(Tensor *A, Tensor *v){ return vectorWise(A,v,div_vec); }
	void inp_divVectorToTensor(Tensor *A, Tensor *v, Tensor *out){ vectorWise(A,v,out,div_vec); }

	Tensor *feq(Tensor *A, Tensor *B){ return elementWise(A,B,eq_tensor); }
	void inp_eq(Tensor *A, Tensor *B, Tensor *out){ elementWise(A,B,out,eq_tensor); }
	Tensor *flt(Tensor *A, Tensor *B){ return elementWise(A,B,lt_tensor); }
	void inp_lt(Tensor *A, Tensor *B, Tensor *out){ elementWise(A,B,out,lt_tensor); }
	Tensor *fgt(Tensor *A, Tensor *B){ return elementWise(A,B,gt_tensor); }
	void inp_gt(Tensor *A, Tensor *B, Tensor *out){ elementWise(A,B,out,gt_tensor); }
	Tensor *fge(Tensor *A, Tensor *B){ return elementWise(A,B,ge_tensor); }
	void inp_ge(Tensor *A, Tensor *B, Tensor *out){ elementWise(A,B,out,ge_tensor); }
	Tensor *fle(Tensor *A, Tensor *B){ return elementWise(A,B,le_tensor); }
	void inp_le(Tensor *A, Tensor *B, Tensor *out){ elementWise(A,B,out,le_tensor); }
	Tensor *fne(Tensor *A, Tensor *B){ return elementWise(A,B,ne_tensor); }
	void inp_ne(Tensor *A, Tensor *B, Tensor *out){ elementWise(A,B,out,ne_tensor); }

	Tensor *fvec_eq(Tensor *A, Tensor *v){ return vectorWise(A,v,eq_vec); }
	void inp_vec_eq(Tensor *A, Tensor *v, Tensor *out){ vectorWise(A,v,out,eq_vec); }
	Tensor *fvec_lt(Tensor *A, Tensor *v){ return vectorWise(A,v,lt_vec); }
	void inp_vec_lt(Tensor *A, Tensor *v, Tensor *out){ vectorWise(A,v,out,lt_vec); }
	Tensor *fvec_gt(Tensor *A, Tensor *v){ return vectorWise(A,v,gt_vec); }
	void inp_vec_gt(Tensor *A, Tensor *v, Tensor *out){ vectorWise(A,v,out,gt_vec); }
	Tensor *fvec_le(Tensor *A, Tensor *v){ return vectorWise(A,v,le_vec); }
	void inp_vec_le(Tensor *A, Tensor *v, Tensor *out){ vectorWise(A,v,out,le_vec); }
	Tensor *fvec_ge(Tensor *A, Tensor *v){ return vectorWise(A,v,ge_vec); }
	void inp_vec_ge(Tensor *A, Tensor *v, Tensor *out){ vectorWise(A,v,out,ge_vec); }
	Tensor *fvec_ne(Tensor *A, Tensor *v){ return vectorWise(A,v,ne_vec); }
	void inp_vec_ne(Tensor *A, Tensor *v, Tensor *out){ vectorWise(A,v,out,ne_vec); }

	Tensor *fscalar_eq(Tensor *A, float flt){ return elementWise(A,NULL,flt,eq_scalar); }
	void inp_scalar_eq(Tensor *A, float flt, Tensor *out){ elementWise(A,NULL,out,flt,eq_scalar); }
	Tensor *fscalar_lt(Tensor *A, float flt){ return elementWise(A,NULL,flt,lt_scalar); }
	void inp_scalar_lt(Tensor *A, float flt, Tensor *out){ elementWise(A,NULL,out,flt,lt_scalar); }
	Tensor *fscalar_gt(Tensor *A, float flt){ return elementWise(A,NULL,flt,gt_scalar); }
	void inp_scalar_gt(Tensor *A, float flt, Tensor *out){ elementWise(A,NULL,out,flt,gt_scalar); }
	Tensor *fscalar_le(Tensor *A, float flt){ return elementWise(A,NULL,flt,le_scalar); }
	void inp_scalar_le(Tensor *A, float flt, Tensor *out){ elementWise(A,NULL,out,flt,le_scalar); }
	Tensor *fscalar_ge(Tensor *A, float flt){ return elementWise(A,NULL,flt,ge_scalar); }
	void inp_scalar_ge(Tensor *A, float flt, Tensor *out){ elementWise(A,NULL,out,flt,ge_scalar); }
	Tensor *fscalar_ne(Tensor *A, float flt){ return elementWise(A,NULL,flt,ne_scalar); }
	void inp_scalar_ne(Tensor *A, float flt, Tensor *out){ elementWise(A,NULL,out,flt,ne_scalar); }

	Tensor *fslice(Tensor *A, Slice *S){ return applySliceFunc(A,S); }

	Tensor *fdot(GPUpy *gpupy, Tensor *A, Tensor *B){ return gpupy->dot(A,B); }
	void inp_dot(GPUpy *gpupy, Tensor *A, Tensor *B, Tensor *out){ gpupy->dot(A,B, out); }
	Tensor *fTdot(GPUpy *gpupy, Tensor *A, Tensor *B){ return gpupy->Tdot(A,B); }
	void inp_Tdot(GPUpy *gpupy, Tensor *A, Tensor *B, Tensor *out){ gpupy->Tdot(A,B, out); }
	Tensor *fdotT(GPUpy *gpupy, Tensor *A, Tensor *B){ return gpupy->dotT(A,B); }
	void inp_dotT(GPUpy *gpupy, Tensor *A, Tensor *B, Tensor *out){ gpupy->dotT(A,B, out); }

	Tensor *fdropout(GPUpy *gpupy, Tensor *A, float dropout_rate){ return gpupy->dropout(A,dropout_rate); }
	void inp_dropout(GPUpy *gpupy, Tensor *A, Tensor *out, float dropout_rate){ gpupy->dropout(A,out,dropout_rate); }

	void fsync_1bit(GPUpy *gpupy, UIntTensor *out1, UIntTensor *out2, UIntTensor *out3, UIntTensor *out4, int layer_idx){ return gpupy->sync_1bit(out1,out2,out3,out4,layer_idx); }
	void fsync_8bit(GPUpy *gpupy, CharTensor *out1, CharTensor *out2, CharTensor *out3, CharTensor *out4, int layer_idx){ return gpupy->sync_8bit(out1,out2,out3,out4,layer_idx); }
	void fsync_16bit(GPUpy *gpupy, UShortTensor *out1, UShortTensor *out2, UShortTensor *out3, UShortTensor *out4, int layer_idx){ return gpupy->sync_16bit(out1,out2,out3,out4,layer_idx); }
	void fsync(GPUpy *gpupy, Tensor *out1, Tensor *out2, Tensor *out3, Tensor *out4, int layer_idx){ return gpupy->sync(out1,out2,out3,out4,layer_idx); }
	void fsynchronize_streams(GPUpy *gpupy, int layer_idx){ gpupy->synchronize_streams(layer_idx); }
	void fcreate_streams(GPUpy *gpupy, int layer_count){ gpupy->createStreams(layer_count); }

	void ftogpu_split(Tensor *out, float *cpu_buffer, int split_idx){ togpu(out,cpu_buffer, split_idx); }

	float fsum(Tensor *A){ return thrust_reduce(A,sum_tensor);}
	float ffmin(Tensor *A){ return thrust_reduce(A,min_tensor);}
	float ffmax(Tensor *A){ return thrust_reduce(A,max_tensor);}

	void inp_RMSProp(Tensor *RMS, Tensor *grad, float RMS_multiplier, float learning_rate, int batch_size)
	{ weightUpdate(RMS, grad, RMS_multiplier, learning_rate, batch_size, RMSProp); }

	int fGPUCount(GPUpy *gpupy){ return gpupy->DEVICE_COUNT; }
	void fenablePeerAccess(GPUpy *gpupy){ gpupy->enablePeerAccess(); }
	void fdisablePeerAccess(GPUpy *gpupy){ gpupy->disablePeerAccess(); }

	void inp_slice_axis(Tensor *A, Tensor *out){ return slice_axis(A, out); }
	void inp_stack_axis(Tensor *A, Tensor *out){ return stack_axis(A, out); }

	float fprint_free_memory(){ return print_free_memory();}

	void fcompress_8bit(GPUpy *gpupy, Tensor *A, float precision, CharTensor *out){ compression_8bit(gpupy->FLT_TABLE_8BIT, A, precision,out); }
	void fdecompress_8bit(GPUpy *gpupy, CharTensor *A, float precision, Tensor *out){ decompression_8bit(gpupy->FLT_TABLE_8BIT, A,precision, out); }
	void fcompress_1bit(Tensor *A_with_errors, Tensor *errors, Tensor *avgPos, Tensor *avgNeg, UIntTensor *out){ compression_1bit(A_with_errors, errors, avgPos, avgNeg, out); }
	void fdecompress_1bit(UIntTensor *quant, Tensor *errors, Tensor *avgPos, Tensor *avgNeg, Tensor *out){ decompression_1bit(quant, errors, avgPos, avgNeg, out); }
	void fcompress_16bit(Tensor *A, UShortTensor *out){ compression_16bit(A, out); }
	void fdecompress_16bit(UShortTensor *A, Tensor *out){ decompression_16bit(A, out); }

	void frow_mean(Tensor *A, Tensor *out){ reduceRow(A, out, NULL,  row_mean); }
	void frow_sum(Tensor *A, Tensor *out){ reduceRow(A, out, NULL,  row_sum); }
	void frow_max(Tensor *A, Tensor *out){ reduceRow(A, out, NULL, row_max); }
	void frow_argmax(Tensor *A, Tensor *out){ reduceRow(A, out, NULL, row_argmax); }
	void frow_max_argmax(Tensor *A, Tensor *out_values, Tensor *out_idxes){ reduceRow(A, out_values, out_idxes, row_max_and_argmax); }

	void ftick(GPUpy *gpupy, char *eventname){ std::string str(eventname); gpupy->tick(str); }
	float ftock(GPUpy *gpupy, char *eventname){ std::string str(eventname); return gpupy->tock(str); }

	void fprintmat(Tensor *A, int start_row, int end_row, int start_col, int end_col){ printmat(A, start_row, end_row, start_col, end_col); }


}
