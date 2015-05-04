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
#include <map>
#include <string>

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

typedef enum Unittype_t
{
	Logistic = 0,
	Rectified_Linear = 1,
	Softmax = 2,
	Linear = 4,
	Double_Rectified_Linear = 8,
	Input = 16
} Unittype_t;

typedef enum DataPropagationType_t
{
	Training = 0,
	Trainerror = 1,
	CVerror = 2
} DataPropagationType_t;


typedef enum ParallelismType_t
{
	None = 0,
	DataParallelism = 1,
	ModelParallelism = 2
} ParallelismType_t;


typedef enum Costfunction_t
{
	Cross_Entropy = 0,
	Squared_Error = 1,
	Root_Squared_Error = 2,
	Misclassification = 4
} Costfunction_t;



class GPUpy
{
public:
	GPUpy();
	int DEVICE_COUNT;
	Tensor *FLT_TABLE_8BIT;

	Tensor *rand(int batchsize, int mapsize, int rows, int cols);
	void rand(Tensor *out);
	Tensor *normal(int batchsize, int mapsize, int rows, int cols, float mean, float std );
	void normal(float mean, float std, Tensor *out);

	Tensor *randn(int batchsize, int mapsize, int rows, int cols);
	void randn(Tensor *out);

	void init(int seed, float* floats_8bit);

	Tensor *dot(Tensor *A, Tensor *B);
	Tensor *Tdot(Tensor *A, Tensor *B);
	Tensor *dotT(Tensor *A, Tensor *B);
	void dot(Tensor *A, Tensor *B, Tensor *out);
	void dotT(Tensor *A, Tensor *B, Tensor *out);
	void Tdot(Tensor *A, Tensor *B, Tensor *out);
	void dot(Tensor *A, Tensor *B, Tensor *out, cublasOperation_t T1, cublasOperation_t T2);
	void enablePeerAccess();
	void disablePeerAccess();
	void synchronize_streams(int layer_idx);
	void sync(Tensor *A, Tensor *out1, Tensor *out2, Tensor *out3, int layer_idx);
	void sync_8bit(CharTensor *A, CharTensor *out1, CharTensor *out2, CharTensor *out3, int layer_idx);
	void sync_1bit(UIntTensor *A, UIntTensor *out1, UIntTensor *out2, UIntTensor *out3, int layer_idx);
	void sync_16bit(UShortTensor *A, UShortTensor *out1, UShortTensor *out2, UShortTensor *out3, int layer_idx);

	Tensor *dropout(Tensor *A, float dropout_rate);
	void dropout(Tensor *A, Tensor *out, float dropout_rate);
	std::vector<cudaStream_t> streams;
	std::vector<cudaStream_t> streams_y;

	void allocateNextAsync(Tensor *batch, float *cpu_buffer, float *pinned_X, Tensor *batch_y, float *cpu_buffer_y, float* pinned_y, int batch_start_idx, int isSplit);
	void replaceCurrentBatch(Tensor *batch_X, Tensor *batch_y, Tensor *buffer_X, Tensor *buffer_y);
	void createStreams(int layer_count);

	void tick();
	void tick(std::string name);
	float tock();
	float tock(std::string name);

	std::vector< std::vector<std::vector<cudaStream_t> > > stream_vectors;
private:
	std::vector<curandGenerator_t> generators;
	std::vector<cublasHandle_t> cublashandles;
	std::map<std::string,cudaEvent_t*> m_dictTickTock;
	std::map<std::string,float> m_dictTickTockCumulative;

};



#endif /* GPUPY_CUH_ */

