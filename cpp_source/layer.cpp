#include <layer.cuh>




using std::cout;
using std::endl;
using std::string;
using std::vector;

Layer::Layer(){}

void Layer::init(int unitcount, int start_batch_size, Unittype_t unit, GPUpy *gpupy, Layer *prev)
{
	next = NULL;
	prev = NULL;
	w_next = NULL;
	b_next = NULL;
	w_rms_next = NULL;
	b_rms_next = NULL;
	w_grad_next = NULL;
	b_grad_next = NULL;

	target = NULL;
	target_Tensor = NULL;
	error = NULL;

	LEARNING_RATE = 0.003;
	RMSPROP_MOMENTUM = 0.9f;
	UNIT_TYPE = unit;
	DROPOUT = 0.5f;
	UNITCOUNT = unitcount;
	BATCH_SIZE = start_batch_size;
	RUNNING_ERROR = 0.0f;
	RUNNING_SAMPLE_SIZE = 0.0f;
	L2 = 15.0f;

	UPDATE_TYPE = RMSProp;
	COST = Misclassification;
	PARALLELISM = None;

	gpu = gpupy;

	if(BATCH_SIZE > 0)
	{
		out = zeros(1,1,BATCH_SIZE, UNITCOUNT);
		bias_activations = ones(1,1,1, BATCH_SIZE);
		activation = zeros(1,1,BATCH_SIZE, UNITCOUNT);
	}
	else
	{
		out = NULL;
		bias_activations = NULL;
		activation = NULL;
	}

	if(prev){ prev->link_with_next_layer(this);}

}

void Layer::link_with_next_layer(Layer *next_layer)
{
	next = next_layer;
	if(next->BATCH_SIZE == 0){ next->BATCH_SIZE = BATCH_SIZE; }
	if(!next->gpu){next->gpu = gpu;}

	//Tensor *w = GPU->uniformSqrtWeight(UNITCOUNT,next_layer->UNITCOUNT);
	Tensor *w = gpu->normal(1,1,UNITCOUNT,next_layer->UNITCOUNT, 0,0.1);
	w_next = w;
	w_grad_next = zeros(1,1,UNITCOUNT,next_layer->UNITCOUNT);
	w_rms_next = zeros(1,1,UNITCOUNT,next_layer->UNITCOUNT);
	//if(PARALLELISM == DataParallelism){w_next_sync = zeros(1,1,UNITCOUNT,next_layer->UNITCOUNT); }

	Tensor *b = zeros(1,1,1,next_layer->UNITCOUNT);
	b_next = b;
	b_grad_next = zeros(1,1,1,next_layer->UNITCOUNT);
	b_rms_next = zeros(1,1,1,next_layer->UNITCOUNT);

	next->out = zeros(1,1,BATCH_SIZE, next->UNITCOUNT);
	next->activation = zeros(1,1,BATCH_SIZE, next->UNITCOUNT);
	next->error = zeros(1,1,BATCH_SIZE, next->UNITCOUNT);
	next->bias_activations = ones(1,1,1, BATCH_SIZE);
	next->prev = this;
}


void Layer::unit_activation(){ unit_activation(true); }
void Layer::unit_activation(bool useDropout)
{
	switch(UNIT_TYPE)
	{
		case Logistic:
			//logistic(out,activation);
			applyFunc(out,NULL,activation,0.0f,logistic);
			break;
		case Rectified_Linear:
			applyFunc(out,NULL,activation,0.0f,rectified_linear);
			break;
		case Softmax:
			softmax(out,out);
			break;
		case Double_Rectified_Linear:
			//doubleRectifiedLinear(out,activation);
			break;
		case Linear:
			//LinearUnit(out, activation);
			break;
		case Input:
			break;
	}

	if(UNIT_TYPE != Softmax)
	{
		if(useDropout)
			gpu->dropout(activation,out,DROPOUT);
		else
			applyFunc(activation,NULL,out,1.0f-DROPOUT,add_scalar);
	}


}

void Layer::activation_gradient()
{

	switch(UNIT_TYPE)
	{
		case Logistic:
			applyFunc(activation,NULL,out,0.0f,logistic_grad);
			break;
		case Rectified_Linear:
			applyFunc(activation,NULL,out,0.0f,gt_tensor);
			break;
		case Double_Rectified_Linear:
			//double_rectified_linear_derivative(activation,out);
			break;
		case Softmax:
			break;
		default:
			throw "Unknown unit";
			break;
	}

}

void Layer::handle_offsize()
{
	if(!prev)
	{
		if(!out){ out = empty(1,1,activation->rows, activation->cols); }
		else if(out->rows != activation->rows)
		{
			cudaFree(out->data);
			free(out);
			out = empty(1,1,activation->rows, activation->cols);
		}
	}
	else
	{
		if(prev->out->rows != out->rows && (!out_offsize || out_offsize->rows != prev->out->rows))
		{
			if(out_offsize)
			{
				cudaFree(out_offsize->data);
				cudaFree(activation_offsize->data);
				cudaFree(error_offsize->data);
				cudaFree(bias_activations_offsize->data);
				cudaFree(target_Tensor_offsize->data);
			}

			out_offsize = empty(1,1,prev->out->rows, UNITCOUNT);
			activation_offsize = empty(1,1,prev->out->rows, UNITCOUNT);
			error_offsize = empty(1,1,prev->out->rows, UNITCOUNT);
			bias_activations_offsize = empty(1,1,1,prev->out->rows);
			target_Tensor_offsize = zeros(1,1,prev->out->rows, UNITCOUNT);
		}


		if(prev->out->rows != out->rows)
		{
			Tensor *swap;
			swap = out; out = out_offsize; out_offsize = swap;
			swap = activation; activation = activation_offsize; activation_offsize = swap;
			swap = error; error = error_offsize; error_offsize = swap;
			swap = bias_activations; bias_activations = bias_activations_offsize; bias_activations_offsize = swap;
			swap = target_Tensor; target_Tensor = target_Tensor_offsize; target_Tensor_offsize = swap;
		}
	}

}


void Layer::dot_switch(Tensor *A, Tensor *B, Tensor *out)
{
	gpu->dot(A,B,out);

	//Tensor *Achar = empty_char(A->rows,A->cols);
	//Tensor *Bchar = empty_char(B->rows,B->cols);
	Tensor *absA = empty(1,1,A->rows,A->cols);
	Tensor *absB = empty(1,1,B->rows,B->cols);

	//abs(A,absA);
	//abs(B,absB);

	//GPU->compression_8bit(A,max(absA),Achar);
	//GPU->compression_8bit(B,max(absB),Bchar);

	//GPU->dot8bit_shared(Achar,Bchar,max(absA),max(absB),out);

	//cudaFree(Achar->char_data);
	//cudaFree(Bchar->char_data);
	cudaFree(absA->data);
	cudaFree(absB->data);

	//free(Achar);
	//free(Bchar);
	free(absA);
	free(absB);

}

void Layer::forward(){ forward(true); }
void Layer::forward(bool useDropout)
{
	handle_offsize();
	if(!prev){  unit_activation(useDropout); next->forward(useDropout); return; }
	if(useDropout){ prev->wait_for_synchronization(); prev->weight_update(); }

	//GPU->dot(prev->out,prev->w_next,out);
	dot_switch(prev->out,prev->w_next,out);
	applyFunc(out,prev->b_next,out,0.0f,add_vec);
    unit_activation(useDropout);

    if(next){ next->forward(useDropout); }
}


void Layer::running_error()
{
	if(!target){ next->running_error(); return;}

	string text = "";

	Tensor *result;
	Tensor *eq;
	float sum_value = 0.0f;

	switch(COST)
	{
		case Misclassification:
			result = argmax(out);
			eq = applyFunc(result,target,gt_tensor);
			sum_value = sum(eq);
			RUNNING_ERROR += (out->rows  - sum_value);
			RUNNING_SAMPLE_SIZE += out->rows;
			break;
		default:
			throw "Unknown cost function!";
			break;
	}

	cudaFree(result->data);
	cudaFree(eq->data);
}



void Layer::backward_errors()
{
	if(!target){ next->backward_errors(); }
	if(target){ applyFunc(activation,target,error,0.0f,sub_tensor); return; }

	if(UNIT_TYPE == Input){ backward_grads(); return; }

	activation_gradient();
	gpu->dotT(next->error, w_next,error);
	applyFunc(error,out,error,0.0f,mul_tensor);
}

void Layer::backward_grads()
{
	gpu->Tdot(activation, next->error, w_grad_next);
	MPI_synchronization_async();
	if(!next->target){ next->backward_grads(); }
	//GPU->dot(next->bias_activations, next->error,b_grad_next);

}

void Layer::MPI_synchronization_async()
{
	/*
	if(PARALLELISM != DataParallelism){ return; }

	int target = gpu->MYRANK +1 == gpu->MPI_SIZE ? 0 : gpu->MYRANK+1;
	int source = gpu->MYRANK-1 == -1 ? gpu->MPI_SIZE-1 : gpu->MYRANK-1;
	*/


	/*
	GPU->compression_8bit(w_grad_next,0.001,w_next_sync_send);
	for (int i = 0; i < GPU->MPI_SIZE - 1; i++)
	{
		MPI_Isend(w_next_sync_send->char_data,w_next_sync_send->size,MPI_CHAR,target,i,MPI_COMM_WORLD, send_request);
		MPI_Irecv(w_next_sync_recv->char_data,w_next_sync_recv->size,MPI_CHAR,source,i,MPI_COMM_WORLD,recv_request);
		target = target +1 == GPU->MPI_SIZE ? 0 : target+1;
		source = source-1 == -1 ? GPU->MPI_SIZE-1 : source-1;
	}
	isSynchronizing = true;
	*/



/*
	for (int i = 0; i < gpu->MPI_SIZE - 1; i++)
	{
		MPI_Isend(w_grad_next->data,w_grad_next->size,MPI_FLOAT,target,i,MPI_COMM_WORLD, send_request);
		MPI_Irecv(w_next_sync->data,w_grad_next->size,MPI_FLOAT,source,i,MPI_COMM_WORLD,recv_request);
		target = target +1 == gpu->MPI_SIZE ? 0 : target+1;
		source = source-1 == -1 ? gpu->MPI_SIZE-1 : source-1;
	}
	isSynchronizing = true;
	*/



}

void Layer::wait_for_synchronization()
{
	/*
	if(target){ return; }
	if(!isSynchronizing){ return; }
	//GPU->tick();
	//MPI_Wait(next->send_request,MPI_STATUS_IGNORE);
	MPI_Wait(recv_request,MPI_STATUS_IGNORE);

	//float secs = GPU->tock()/1000.0f;
	//cout << w_next_sync->bytes/1024./1024./1024./secs << " GB/s" << endl;
	//printdim(w_next_sync);

	//GPU->decompression_8bit(w_next_sync_recv,0.001,w_next_sync);
	add(w_next_sync,w_grad_next,w_grad_next);
	isSynchronizing = false;
	*/
}

void Layer::weight_update()
{
	if(target){ return; }

	//next->weight_update();

	weightUpdate(w_rms_next,w_grad_next,RMSPROP_MOMENTUM,LEARNING_RATE,out->rows, UPDATE_TYPE);
	//limit_magnitude();

}

void Layer::limit_magnitude()
{

	//square(w_next,w_grad_next);
	Tensor *temp = ones(1,1,w_grad_next->cols,1);
	Tensor *sums = gpu->dot(w_grad_next,temp);
	//renormalizeWeights(w_next,sums,L2);
	cudaFree(temp->data);
	cudaFree(sums->data);
	free(temp);
	free(sums);

}

void Layer::print_error(string message)
{
	if(!target){ next->print_error(message); return;}

	cout << message << RUNNING_ERROR/RUNNING_SAMPLE_SIZE << endl;
	RUNNING_ERROR = 0.0f;
	RUNNING_SAMPLE_SIZE = 0.0f;
}

void Layer::set_hidden_dropout(float dropout)
{
	if(!next){ return; }
	next->DROPOUT = dropout;
	next->set_hidden_dropout(dropout);
}

void Layer::learning_rate_decay(float decay_rate)
{
	if(!next){ return; }
	next->LEARNING_RATE *= decay_rate;
	next->learning_rate_decay(decay_rate);
}

void Layer::dropout_decay()
{
	if(!prev){ cout << "Decaying dropout!" << endl; }
	if(!next){ return;}

	cout << "Setting dropout from " << DROPOUT << " to " << DROPOUT/2.0f << endl;
	DROPOUT /= 2.0f;
	next->dropout_decay();
}

Layer::~Layer()
{
	cout << "destruct" << endl;
}


