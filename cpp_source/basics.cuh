/*
 * basics.cuh
 *
 *  Created on: Mar 21, 2015
 *      Author: tim
 */

#ifndef BASICS_CUH_
#define BASICS_CUH_
#include <Tensor.cuh>

#define THREADS_PER_BLOCKS (512)

Tensor *empty(int batches, int maps, int rows, int cols);
Tensor *zeros(int batches, int maps, int rows, int cols);
Tensor *ones(int batches, int maps, int rows, int cols);
Tensor *fill_with_number(Tensor *A, float number);

void togpu(Tensor *out, float *cpu_buffer);

Tensor *tocpu(Tensor *A, float *cpu_buffer);
Tensor *T(Tensor *A);
void T(Tensor *A, Tensor *out, int rows, int cols);

Tensor *to_col_major(Tensor *A);
void to_col_major(Tensor *A, Tensor *out);
Tensor *to_row_major(Tensor *A);

Tensor *add(Tensor *A, Tensor *B);
void add(Tensor *A, Tensor *B, Tensor *out);
Tensor *sub(Tensor *A, Tensor *B);
void sub(Tensor *A, Tensor *B, Tensor *out);
Tensor *mul(Tensor *A, Tensor *B);
void mul(Tensor *A, Tensor *B, Tensor *out);
Tensor *div(Tensor *A, Tensor *B);
void div(Tensor *A, Tensor *B, Tensor *out);





#endif /* BASICS_CUH_ */

