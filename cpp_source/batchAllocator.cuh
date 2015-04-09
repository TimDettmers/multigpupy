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
	void allocateNextAsync(Tensor *batch, float *cpu_buffer, Tensor *batch_y, float *cpu_buffer_y);
	void replaceCurrentBatch();

	std::vector<cudaStream_t> streams;
	std::vector<cudaStream_t> streams_y;
	int DEVICE_COUNT;


};


#endif /* BATCHALLOCATOR_CUH_ */
