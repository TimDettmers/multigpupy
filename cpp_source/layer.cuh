/*
 * layer.cuh
 *
 *  Created on: Apr 9, 2015
 *      Author: tim
 */

#ifndef LAYER_CUH_
#define LAYER_CUH_
#include <gpupy.cuh>
#include <Tensor.cuh>
#include <basics.cuh>

class Layer
{
public:
	Tensor *w_grad_next;
	Tensor *b_grad_next;
	Layer *next;
	Layer *prev;
	Tensor *w_next;
	Tensor *b_next;

	Tensor *w_rms_next;
	Tensor *b_rms_next;

	Tensor *bias_activations;
	Tensor *out;
	Tensor *error;
	Tensor *activation;

	Tensor *out_offsize;
	Tensor *activation_offsize;
	Tensor *error_offsize;
	Tensor *bias_activations_offsize;
	Tensor *target_Tensor_offsize;

	Tensor *target;
	Tensor *target_Tensor;

	GPUpy *gpu;

	float LEARNING_RATE;
	float MOMENTUM;
	float RMSPROP_MOMENTUM;
	float RUNNING_ERROR;
	float RUNNING_SAMPLE_SIZE;
	float L2;
    Unittype_t UNIT_TYPE;
	Costfunction_t COST;
	float DROPOUT;
	int UNITCOUNT;
	int BATCH_SIZE;

	bool isSynchronizing;

	weightUpdate_t UPDATE_TYPE;

	ParallelismType_t PARALLELISM;

	virtual ~Layer();
	Layer();

	virtual void forward();
	virtual void forward(bool useDropout);
	virtual void running_error();
	virtual void backward_errors();
	virtual void backward_grads();
	virtual void print_error(std::string message);
	virtual void weight_update();

	virtual void MPI_synchronization_async();
	virtual void wait_for_synchronization();

	virtual void limit_magnitude();

	virtual void link_with_next_layer(Layer *next_layer);
	virtual void init(int unitcount, int start_batch_size, Unittype_t unit, GPUpy *gpupy, Layer *prev);
	virtual void set_hidden_dropout(float dropout);

	virtual void dropout_decay();
	virtual void learning_rate_decay(float decay_rate);

	virtual void dot_switch(Tensor *A, Tensor *B, Tensor *out);



private:
	virtual void unit_activation();
	virtual void unit_activation(bool useDropout);
	virtual void activation_gradient();
	void handle_offsize();


};



#endif /* LAYER_CUH_ */
