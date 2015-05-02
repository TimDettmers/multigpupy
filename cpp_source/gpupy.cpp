#include <gpupy.cuh>
#include <basics.cuh>
#include <cudaKernels.cuh>

using std::cout;
using std::endl;

GPUpy::GPUpy(){}

void GPUpy::init(int seed, float *floats_8bit)
{
	DEVICE_COUNT = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&DEVICE_COUNT));


	Tensor *temp = zeros(1,1,1,128);
	togpu(temp, floats_8bit);

	FLT_TABLE_8BIT = temp;

	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		curandGenerator_t gen;

		CURAND_CHECK_RETURN(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A));
		CURAND_CHECK_RETURN(curandSetPseudoRandomGeneratorSeed(gen, seed));
		CURAND_CHECK_RETURN(curandSetGeneratorOffset(gen, 100));

		generators.push_back(gen);
		cublasHandle_t handle;
		CUBLAS_CHECK_RETURN(cublasCreate_v2(&handle));
		cublashandles.push_back(handle);

		CUDA_CHECK_RETURN(cudaSetDevice(i));
		cudaStream_t s;
		//CUDA_CHECK_RETURN(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
		CUDA_CHECK_RETURN(cudaStreamCreate(&s));
		streams.push_back(s);
		cudaStream_t s_y;
		//CUDA_CHECK_RETURN(cudaStreamCreateWithFlags(&s_y, cudaStreamNonBlocking));
		CUDA_CHECK_RETURN(cudaStreamCreate(&s_y));
		streams_y.push_back(s_y);

	}

	CUDA_CHECK_RETURN(cudaSetDevice(0));

	createStreams(1);

	CURRENT_SYNC_IDX = 0;
	IS_SYNCHRONIZING = 0;

}

void GPUpy::createStreams(int layer_count)
{

	/*

	for(int i = 0; i < stream_vectors.size(); i++)
		for(int j = 0; j < stream_vectors[i].size(); j++)
		{
			CUDA_CHECK_RETURN(cudaSetDevice(i));
			for(int k = 0; k < stream_vectors[i][j].size(); k++)
				CUDA_CHECK_RETURN(cudaStreamDestroy(stream_vectors[i][j][k]));
		}


	stream_vectors.clear();
	*/

	for(int j = 0; j < layer_count; j++)
	{
		std::vector<std::vector<cudaStream_t> > layer;
		for(int i = 0; i < DEVICE_COUNT; i++)
		{
			CUDA_CHECK_RETURN(cudaSetDevice(i));
			std::vector<cudaStream_t> vec;
			for(int k = 0; k < DEVICE_COUNT; k++)
			{
				cudaStream_t s;
				//CUDA_CHECK_RETURN(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
				CUDA_CHECK_RETURN(cudaStreamCreate(&s));
				vec.push_back(s);
			}
			layer.push_back(vec);
		}
		stream_vectors.push_back(layer);
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));

}


Tensor *GPUpy::rand(int batchsize, int mapsize, int rows, int cols)
{ Tensor *out = empty(batchsize, mapsize, rows, cols); rand(out); return out; }
void GPUpy::rand(Tensor *out)
{
	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		CURAND_CHECK_RETURN(curandGenerateUniform(generators[i], out->data_gpus[i],out->size_gpus[i]));
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *GPUpy::normal(int batchsize, int mapsize, int rows, int cols, float mean, float std)
{ Tensor *out = empty(batchsize, mapsize, rows, cols); normal(mean, std, out); return out; }
void GPUpy::normal(float mean, float std, Tensor *out)
{
	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaSetDevice(i));
		CURAND_CHECK_RETURN(curandGenerateNormal(generators[i], out->data_gpus[i],out->size_gpus[i],mean,std));
	}
	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

Tensor *GPUpy::randn(int batchsize, int mapsize, int rows, int cols){ return normal(batchsize,mapsize, rows, cols, 0.0f,1.0f); }
void GPUpy::randn(Tensor *out){ normal(0.0f,1.0f,out); }

Tensor *GPUpy::dot(Tensor *A, Tensor *B){ Tensor *out = empty(1,1,A->rows,B->cols); dot(A,B,out,CUBLAS_OP_N,CUBLAS_OP_N); return out; }
Tensor *GPUpy::Tdot(Tensor *A, Tensor *B){ Tensor *out = empty(1,1,A->rows,B->cols); dot(A,B,out,CUBLAS_OP_T,CUBLAS_OP_N); return out; }
Tensor *GPUpy::dotT(Tensor *A, Tensor *B){ Tensor *out = empty(1,1,A->rows,B->cols); dot(A,B,out,CUBLAS_OP_N,CUBLAS_OP_T); return out; }
void GPUpy::dot(Tensor *A, Tensor *B, Tensor *out){ dot(A,B,out,CUBLAS_OP_N,CUBLAS_OP_N); }
void GPUpy::dotT(Tensor *A, Tensor *B, Tensor *out){ dot(A,B,out,CUBLAS_OP_N,CUBLAS_OP_T); }
void GPUpy::Tdot(Tensor *A, Tensor *B, Tensor *out){ dot(A,B,out,CUBLAS_OP_T,CUBLAS_OP_N); }
void GPUpy::dot(Tensor *A, Tensor *B, Tensor *out, cublasOperation_t T1, cublasOperation_t T2)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		int A_rows = A->shape_gpus[i][2], A_cols = A->shape_gpus[i][3], B_rows = B->shape_gpus[i][2], B_cols = B->shape_gpus[i][3];
		if (T1 == CUBLAS_OP_T)
		{
			A_rows = A->shape_gpus[i][3];
			A_cols = A->shape_gpus[i][2];
		}
		if (T2 == CUBLAS_OP_T)
		{
			B_rows = B->shape_gpus[i][3];
			B_cols = B->shape_gpus[i][2];
		}

		assert(A->shape_gpus[i][1] == 1 && "Tensors dot product is not supported.");
		assert(A->shape_gpus[i][0] == 1 && "Tensors dot product is not supported.");

		if(A_cols != B_rows)
		{
			cout << "A rows: " << A_rows << " vs. " <<  "A cols: " <<B_cols << endl;
			cout << "B rows: " << B_rows << " vs. " <<  "B cols: " <<B_cols << endl;
			cout << "out rows: " << out->shape_gpus[i][2] << " vs. " <<  "out cols: " << out->shape_gpus[i][3] << endl;
		}
		assert(A_cols == B_rows);
		if(out->shape_gpus[i][2] != A_rows || out->shape_gpus[i][3] != B_cols)
		{
			cout << "A rows: " << A_rows << " vs. " <<  "A cols: " <<B_cols << endl;
			cout << "B rows: " << B_rows << " vs. " <<  "B cols: " <<B_cols << endl;
			cout << "out rows: " << out->shape_gpus[i][2] << " vs. " <<  "out cols: " << out->shape_gpus[i][3] << endl;
		}
		assert(out->shape_gpus[i][2] == A_rows && out->shape_gpus[i][3] == B_cols);


		CUDA_CHECK_RETURN(cudaSetDevice(i));

		//print_tensor_shape(A);
		//print_tensor_shape(B);
		//print_tensor_shape(out);

		CUBLAS_CHECK_RETURN(cublasSgemm(cublashandles[i], T1, T2, A_rows, B_cols,
				A_cols, &alpha, A->data_gpus[i], A->shape_gpus[i][2], B->data_gpus[i], B->shape_gpus[i][2], &beta,
				out->data_gpus[i], out->shape_gpus[i][2]));
	}

	CUDA_CHECK_RETURN(cudaSetDevice(0));


}


Tensor *GPUpy::dropout(Tensor *A, float dropout_rate)
{
	Tensor *out = empty(A->batches, A->maps, A->rows, A->cols);

	dropout(A, out, dropout_rate);
	return out;
}

void GPUpy::dropout(Tensor *A, Tensor *out, float dropout_rate)
{
	rand(out);
	elementWise(A, NULL, out, dropout_rate, dropout_tensor);
}



void GPUpy::enablePeerAccess()
{
	for(int gpu1 = 0; gpu1 < DEVICE_COUNT; gpu1++)
		for(int gpu2 = 0; gpu2 < DEVICE_COUNT; gpu2++)
			if(gpu1!=gpu2)
			{
				CUDA_CHECK_RETURN(cudaSetDevice(gpu1));
				CUDA_CHECK_RETURN(cudaDeviceEnablePeerAccess(gpu2,0));
			}

	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

void GPUpy::disablePeerAccess()
{
	for(int gpu1 = 0; gpu1 < DEVICE_COUNT; gpu1++)
		for(int gpu2 = 0; gpu2 < DEVICE_COUNT; gpu2++)
			if(gpu1!=gpu2)
			{
				CUDA_CHECK_RETURN(cudaSetDevice(gpu1));
				CUDA_CHECK_RETURN(cudaDeviceDisablePeerAccess(gpu2));
			}

	CUDA_CHECK_RETURN(cudaSetDevice(0));
}

template <typename T>
void async_sync_template(GPUpy * gpupy, T *A, T *out1, T *out2, T *out3, int layer_idx)
{
	std::vector<T*> out;
	out.push_back(A);
	if(out1) out.push_back(out1);
	if(out2) out.push_back(out2);
	if(out3) out.push_back(out3);
	int idx = 0;

	//left-right transfer across PCIe switches
	//this is the fastest transfer method for multi-GPU setups on non-specialized hardware
	//this transfer is made exactly like the matrix cross product where left and right transfers are left and right arrows
	//tick();
	gpupy->IS_SYNCHRONIZING = 1;
	for(int transfer_round = 1; transfer_round < gpupy->DEVICE_COUNT; transfer_round++)
	{
		//right transfer
		for(int right_idx = 0; right_idx < gpupy->DEVICE_COUNT-transfer_round; right_idx++)
		{
			CUDA_CHECK_RETURN(cudaMemcpyAsync(out[right_idx+transfer_round]->data_gpus[right_idx+transfer_round], A->data_gpus[right_idx],A->bytes_gpus[right_idx],cudaMemcpyDefault, gpupy->stream_vectors[layer_idx][right_idx][right_idx+transfer_round]));
			//cout << "right from " << right_idx+transfer_round << " to " << right_idx << " with stream " << layer_idx << "x" << right_idx << right_idx+transfer_round << endl;
		}



		//left transfer
		for(int left_idx = transfer_round; left_idx < gpupy->DEVICE_COUNT; left_idx++)
		{
			idx = (gpupy->DEVICE_COUNT)-transfer_round;
			CUDA_CHECK_RETURN(cudaMemcpyAsync(out[idx]->data_gpus[left_idx-transfer_round], A->data_gpus[left_idx],A->bytes_gpus[left_idx],cudaMemcpyDefault, gpupy->stream_vectors[layer_idx][left_idx][left_idx-transfer_round]));
			//cout << "left from " << idx <<  " and " << left_idx-transfer_round << " to " << left_idx << " with stream " << layer_idx << "x" << left_idx << left_idx-transfer_round << endl;
		}
	}
	//tick();

}


void GPUpy::async_sync_1bit(UIntTensor *A, UIntTensor *out1, UIntTensor *out2, UIntTensor *out3, int layer_idx)
{ async_sync_template<UIntTensor>(this, A, out1, out2, out3, layer_idx); }
void GPUpy::async_sync_8bit(CharTensor *A, CharTensor *out1, CharTensor *out2, CharTensor *out3, int layer_idx)
{ async_sync_template<CharTensor>(this, A, out1, out2, out3, layer_idx); }
void GPUpy::async_sync_16bit(UShortTensor *A, UShortTensor *out1, UShortTensor *out2, UShortTensor *out3, int layer_idx)
{ async_sync_template<UShortTensor>(this, A, out1, out2, out3, layer_idx); }
void GPUpy::async_sync(Tensor *A, Tensor *out1, Tensor *out2, Tensor *out3, int layer_idx)
{ async_sync_template<Tensor>(this, A, out1, out2, out3, layer_idx); }


void GPUpy::synchronize_streams(int layer_idx)
{

	for(int i = 0; i < DEVICE_COUNT; i++)
		for(int j = 0; j < DEVICE_COUNT; j++)
			CUDA_CHECK_RETURN(cudaStreamSynchronize(stream_vectors[layer_idx][i][j]));


	//CUDA_CHECK_RETURN(cudaStreamSynchronize(stream_vectors[layer_idx][1][0]));
	//CUDA_CHECK_RETURN(cudaStreamSynchronize(stream_vectors[layer_idx][0][1]));
	CURRENT_SYNC_IDX +=1;

}




void GPUpy::allocateNextAsync(Tensor *batch, float *cpu_buffer, float *pinned_X, Tensor *batch_y, float *cpu_buffer_y, float* pinned_y, int batch_start_idx, int isSplit)
{
	//memcpy(pinned_X, cpu_buffer, batch->bytes);
	//memcpy(pinned_y, cpu_buffer_y, batch_y->bytes);

	//to_col_major_pinned(&cpu_buffer[batch_start_idx*batch->cols], pinned_X,1,1,batch->rows, batch->cols);
	//to_col_major_pinned(&cpu_buffer_y[batch_start_idx*batch_y->cols], pinned_y,1,1,batch_y->rows, batch_y->cols);
	int offset = 0;
	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		if(isSplit == -1)
		{
			CUDA_CHECK_RETURN(cudaMemcpyAsync(batch->data_gpus[i],&cpu_buffer[batch_start_idx*batch->cols],batch->bytes_gpus[i],cudaMemcpyDefault, streams[i]));
			CUDA_CHECK_RETURN(cudaMemcpyAsync(batch_y->data_gpus[i],&cpu_buffer_y[batch_start_idx*batch_y->cols],batch_y->bytes_gpus[i],cudaMemcpyDefault, streams_y[i]));
		}
		else
		{
			CUDA_CHECK_RETURN(cudaMemcpyAsync(batch->data_gpus[i],&cpu_buffer[(batch_start_idx+offset)*batch->cols],batch->bytes_gpus[i],cudaMemcpyDefault, streams[i]));
			CUDA_CHECK_RETURN(cudaMemcpyAsync(batch_y->data_gpus[i],&cpu_buffer_y[(batch_start_idx+offset)*batch_y->cols],batch_y->bytes_gpus[i],cudaMemcpyDefault, streams_y[i]));

			offset += batch->shape_gpus[i][2];

		}
	}
}

void GPUpy::replaceCurrentBatch()
{


	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[i]));
		CUDA_CHECK_RETURN(cudaStreamSynchronize(streams_y[i]));
	}


}

void GPUpy::replaceCurrentBatch(Tensor *batch_X, Tensor *batch_y, Tensor *buffer_X, Tensor *buffer_y)
{


	for(int i = 0; i < DEVICE_COUNT; i++)
	{
		CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[i]));
		CUDA_CHECK_RETURN(cudaStreamSynchronize(streams_y[i]));
	}


	to_col_major(buffer_X, batch_X);
	to_col_major(buffer_y, batch_y);

}


void GPUpy::tick()
{
	tick("default");
}
void GPUpy::tick(std::string name)
{
	if (m_dictTickTock.count(name) > 0)
	{
		if (m_dictTickTockCumulative.count(name) > 0)
		{
			m_dictTickTockCumulative[name] += ::tock(m_dictTickTock[name],
					0.0f);
			m_dictTickTock.erase(name);
		} else
		{
			m_dictTickTockCumulative[name] = ::tock(m_dictTickTock[name], 0.0f);
			m_dictTickTock.erase(name);
		}
	} else
	{
		m_dictTickTock[name] = ::tick();
	}
}
float GPUpy::tock(){ return tock("default"); }
float GPUpy::tock(std::string name)
{
	if (m_dictTickTockCumulative.count(name) > 0)
	{
		::tock("<<<Cumulative>>>: " + name, m_dictTickTockCumulative[name]);
		float value = m_dictTickTockCumulative[name];
		m_dictTickTockCumulative.erase(name);
		return value;
	}
	else
	{
		if (m_dictTickTock.count(name) == 0)
			cout << "Error for name: " << name << endl;
		assert(("No tick event was registered for the name" + name, m_dictTickTock.count(name) > 0));
		float value = ::tock(m_dictTickTock[name], name);
		m_dictTickTock.erase(name);
		return value;
	}
}


