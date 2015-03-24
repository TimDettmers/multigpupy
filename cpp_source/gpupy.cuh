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



#define CURAND_CHECK_RETURN(value) {											\
	curandStatus_t _m_cudaStat = value;										\
	if (_m_cudaStat != CURAND_STATUS_SUCCESS) {										\
		fprintf(stderr, "Error %i at line %d in file %s\n",					\
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
private:
	std::vector<curandGenerator_t> generators;

};



#endif /* GPUPY_CUH_ */

