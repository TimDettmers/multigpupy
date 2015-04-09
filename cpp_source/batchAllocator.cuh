/*
 * batchAllocator.cuh
 *
 *  Created on: Apr 8, 2015
 *      Author: tim
 */

#ifndef BATCHALLOCATOR_CUH_
#define BATCHALLOCATOR_CUH_
#include <Tensor.cuh>
#include <basics.cuh>

class BatchAllocator
{
public:
	BatchAllocator();
	void allocateNextAsync(Tensor *batch, float *cpu_buffer);
	void replaceCurrentBatch(Tensor *current_batch, Tensor *next_batch);

	std::vector<cudaStream_t> streams;
	int DEVICE_COUNT;


};


#endif /* BATCHALLOCATOR_CUH_ */
