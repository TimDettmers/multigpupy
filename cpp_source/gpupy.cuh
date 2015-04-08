/*
 * gpupy.cuh
 *
 *  Created on: Mar 22, 2015
 *      Author: tim
 */

#ifndef GPUPY_CUH_
#define GPUPY_CUH_
#include <Tensor.cuh>
#include <curand.h>
#include <cublas_v2.h>




#define CURAND_CHECK_RETURN(value) {											\
	curandStatus_t _m_cudaStat = value;										\
	if (_m_cudaStat != CURAND_STATUS_SUCCESS) {										\
		fprintf(stderr, "CURAND error %i at line %d in file %s\n",					\
				_m_cudaStat, __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CUBLAS_CHECK_RETURN(value) {											\
	cublasStatus_t _m_cudaStat = value;										\
	if (_m_cudaStat != CUBLAS_STATUS_SUCCESS) {										\
		fprintf(stderr, "CUBLAS error %i at line %d in file %s\n",					\
				_m_cudaStat, __LINE__, __FILE__);		\
		exit(1);															\
	} }

class GPUpy
{
public:
	GPUpy();
	int DEVICE_COUNT;

	Tensor *rand(int batchsize, int mapsize, int rows, int cols);
	void rand(Tensor *out);
	Tensor *normal(int batchsize, int mapsize, int rows, int cols, float mean, float std );
	void normal(float mean, float std, Tensor *out);

	Tensor *randn(int batchsize, int mapsize, int rows, int cols);
	void randn(Tensor *out);

	void init(int seed);

	Tensor *dot(Tensor *A, Tensor *B);
	Tensor *Tdot(Tensor *A, Tensor *B);
	Tensor *dotT(Tensor *A, Tensor *B);
	void dot(Tensor *A, Tensor *B, Tensor *out);
	void dotT(Tensor *A, Tensor *B, Tensor *out);
	void Tdot(Tensor *A, Tensor *B, Tensor *out);
	void dot(Tensor *A, Tensor *B, Tensor *out, cublasOperation_t T1, cublasOperation_t T2);

private:
	std::vector<curandGenerator_t> generators;
	std::vector<cublasHandle_t> cublashandles;

};



#endif /* GPUPY_CUH_ */

