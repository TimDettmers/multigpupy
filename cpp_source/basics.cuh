/*
 * basics.cuh
 *
 *  Created on: Mar 21, 2015
 *      Author: tim
 */

#ifndef BASICS_CUH_
#define BASICS_CUH_
#include <Tensor.cuh>
#include <thrust/reduce.h>
#define THREADS_PER_BLOCKS (512)



Slice *emptySlice();
Tensor *empty_like(Tensor *A);
Tensor *empty(int batches, int maps, int rows, int cols);
Tensor *empty(int batches, int maps, int rows, int cols, int split_idx);
CharTensor *empty_char_like(Tensor *A);
CharTensor *empty_char(int batches, int maps, int rows, int cols);
CharTensor *empty_char(int batches, int maps, int rows, int cols, int split_idx);
UIntTensor *empty_uint_like(Tensor *A);
UIntTensor *empty_uint(int batches, int maps, int rows, int cols);
UIntTensor *empty_uint(int batches, int maps, int rows, int cols, int split_idx);

UShortTensor *empty_ushort_like(Tensor *A);
UShortTensor *empty_ushort(int batches, int maps, int rows, int cols);
UShortTensor *empty_ushort(int batches, int maps, int rows, int cols, int split_axis);

float *empty_pinned(int batches, int maps, int rows, int cols,float *cpu_buffer);
Tensor *zeros(int batches, int maps, int rows, int cols);
Tensor *zeros(int batches, int maps, int rows, int cols, int split_axis);
Tensor *ones(int batches, int maps, int rows, int cols);

int *get_split_shape(int batches, int maps, int rows, int cols,int split_axis,int gpuidx);

void togpu(Tensor *out, float *cpu_buffer);
void togpu(Tensor *out, float *cpu_buffer, int split_axis);

void print_slice(Slice *S);
void print_shape(int *shape);
float print_free_memory();
void print_tensor_shape(Tensor *A);

Tensor *tocpu(Tensor *A, float *cpu_buffer);
Tensor *T(Tensor *A);
void T(Tensor *A, Tensor *out, int rows, int cols);
Tensor *softmax(Tensor *A);
void softmax(Tensor *A, Tensor *out);
Tensor *argmax(Tensor *A);
void argmax(Tensor *A, Tensor *out);

Tensor *to_col_major(Tensor *A);
void to_col_major(Tensor *A, Tensor *out);
Tensor *to_row_major(Tensor *A);

Tensor *vectorWise(Tensor *A, Tensor *B, Operation_t ops);
Tensor *vectorWise(Tensor *A, Tensor *B, float flt, Operation_t ops);
void vectorWise(Tensor *A, Tensor *B, Tensor *out, Operation_t ops);
void vectorWise(Tensor *A, Tensor *B, Tensor *out, float flt, Operation_t ops);

Tensor *elementWise(Tensor *A, Tensor *B, Operation_t ops);
Tensor *elementWise(Tensor *A, Tensor *B, float flt, Operation_t ops);
void elementWise(Tensor *A, Tensor *B, Tensor *out, Operation_t ops);
void elementWise(Tensor *A, Tensor *B, Tensor *out, float flt, Operation_t ops);

Tensor *applySliceFunc(Tensor *A, Slice *S);
void applySliceFunc(Tensor *A, Slice *S, Tensor *out);

float thrust_reduce(Tensor *A, Operation_t strategy);

void weightUpdate(Tensor *RMS, Tensor *grad, float RMS_multiplier, float learning_rate, int batch_size, weightUpdate_t strategy);

void slice_axis(Tensor *A, Tensor *out);
void stack_axis(Tensor *A, Tensor *out);

void compression_8bit(Tensor *tbl_flt, Tensor *A, float precision,  CharTensor *out);
void decompression_8bit(Tensor *tbl_flt, CharTensor *A, float precision,  Tensor *out);
void compression_1bit(Tensor *A_with_errors, Tensor *errors, Tensor *avgPos, Tensor *avgNeg, UIntTensor *out);
void decompression_1bit(UIntTensor *quant, Tensor *errors, Tensor *avgPos, Tensor *avgNeg, Tensor *out);
void compression_16bit(Tensor *A, UShortTensor *out);
void decompression_16bit(UShortTensor *A, Tensor *out);

void reduceRow(Tensor *A, Tensor *out, Operation_t ops);

cudaEvent_t* tick();
float tock(cudaEvent_t* startstop);
float tock(cudaEvent_t* startstop, std::string text);
float tock(std::string text, float tocks);
float tock(cudaEvent_t* startstop, float tocks);

#endif /* BASICS_CUH_ */

