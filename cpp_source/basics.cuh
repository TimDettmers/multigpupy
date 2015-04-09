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
Tensor *empty(int batches, int maps, int rows, int cols);
Tensor *empty_pinned(int batches, int maps, int rows, int cols,float *cpu_buffer);
Tensor *zeros(int batches, int maps, int rows, int cols);
Tensor *ones(int batches, int maps, int rows, int cols);
Tensor *fill_with_number(Tensor *A, float number);

void togpu(Tensor *out, float *cpu_buffer);

Tensor *tocpu(Tensor *A, float *cpu_buffer);
Tensor *T(Tensor *A);
void T(Tensor *A, Tensor *out, int rows, int cols);
Tensor *softmax(Tensor *A);
void softmax(Tensor *A, Tensor *out);

Tensor *to_col_major(Tensor *A);
void to_col_major(Tensor *A, Tensor *out);
Tensor *to_row_major(Tensor *A);

Tensor *applyFunc(Tensor *A, Tensor *B, Operation_t ops);
Tensor *applyFunc(Tensor *A, Tensor *B, float flt, Operation_t ops);
void applyFunc(Tensor *A, Tensor *B, Tensor *out, Operation_t ops);
void applyFunc(Tensor *A, Tensor *B, Tensor *out, float flt, Operation_t ops);

void synchronize(Tensor *A, Tensor *out, int myid, int copyid, cudaStream_t stream,Operation_t ops);

Tensor *applySliceFunc(Tensor *A, Slice *S);
void applySliceFunc(Tensor *A, Slice *S, Tensor *out);
int sliceDimHelper(int dim, int start, int stop);
void rearrageSlice(Slice *S, Tensor *A);

float sum(Tensor *A);
float max(Tensor *A);
float min(Tensor *A);




#endif /* BASICS_CUH_ */

