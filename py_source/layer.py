'''
Created on Apr 9, 2015

@author: tim
'''
from gpupy import array, p_gpupy
import numpy as np
import util as u
from library_interface import lib
import ctypes as ct
import gpupy as gpu
import logging


'''
strategy pattern for torch7 API
'''
class ActivationFunc(object):
    def __init__(self, dropout, gpu_func, gpu_func_grad):    
        self.dropout = dropout   
        self.gpu_func = gpu_func
        self.gpu_func_grad = gpu_func_grad
    def activation(self, previous_output, my_activation, my_output, useDropout): 
        self.gpu_func(previous_output, my_activation);   
        if useDropout and self.dropout > 0.0: gpu.dropout(my_activation, self.dropout, my_output)
        else: gpu.mul(my_activation, 1.0-self.dropout, my_output)
        
    def grad(self, my_activation, my_output): return self.gpu_func_grad(my_activation, my_output)  
    
class Logistic(ActivationFunc): 
    def __init__(self):  
        super(Logistic, self).__init__(0.5, gpu.logistic, gpu.logistic_grad)
            
class ReLU(ActivationFunc):      
    def __init__(self):  
        super(ReLU, self).__init__(0.5, gpu.ReLU, gpu.ReLU_grad)      
        
class Input(ActivationFunc): 
    def __init__(self):  
        super(Input, self).__init__(0.2, gpu.linear, gpu.linear)
        
class Linear(ActivationFunc): 
    def __init__(self):  
        super(Linear, self).__init__(0.0, gpu.linear, gpu.linear)
    def activation(self, previous_output, my_activation, my_output, useDropout): 
        self.gpu_func(previous_output, my_output); 
        
class Softmax(ActivationFunc):
    def __init__(self):   
        super(Softmax, self).__init__(0.0, gpu.softmax, gpu.linear)
    def activation(self, previous_output, my_activation, my_output, useDropout): 
        self.gpu_func(previous_output, my_output); 

class Layer(object):
    def __init__(self, unitcount=0, activation_function=Input()):
        self.p_layer = lib.funcs.fLayer()
        self.w_next = None
        self.activation = None
        self.activation_offsize = None         
        self.funcs = activation_function
        self.unitcount = unitcount
        self.next_layer = None
        self.prev_layer = None
        self.id = 0
        self.target = None
        self.current_error = []
        self.current_SE = []
        self.error_epochs = []
        self.confidence_interval_epochs = []
        self.config = {'learning_rate' : 0.03,
                       'momentum' : 0.9,
                       'input_dropout': 0.2,
                       'dropout' : 0.5
                       }        
        self.logger = None
        
    def add(self,next_layer, logger=None):  
        if self.next_layer:    
            self.next_layer.add(next_layer,self.logger)
            return
        
        if type(next_layer) is Layer:            
            self.next_layer = next_layer
            next_layer.prev_layer = self
            self.next_layer.logger = self.logger   
            next_layer.id = self.id +1
            
        else:
            self.funcs = next_layer
    
    def create_weights(self):
        if self.next_layer:
            self.w_next = gpu.array(u.create_uniform_rdm_weight(self.unitcount,self.next_layer.unitcount))
            self.b_next = gpu.zeros((1, self.next_layer.unitcount))
            self.m_next = gpu.zeros((self.unitcount, self.next_layer.unitcount))
            self.w_grad_next = gpu.zeros((self.unitcount, self.next_layer.unitcount))
            if self.next_layer: self.next_layer.create_weights()
        
    def create_buffers(self, batch_size):
        self.activation = gpu.empty((batch_size,self.unitcount))
        self.out = gpu.empty((batch_size,self.unitcount))
        self.error = gpu.empty((batch_size,self.unitcount))
        self.bias = gpu.ones((1,batch_size))
        
    def handle_offsize(self, batch_size):
        if self.activation_offsize == None:
            print 'swap'
            self.activation_offsize = gpu.empty((batch_size,self.unitcount))
            self.out_offsize = gpu.empty((batch_size,self.unitcount))
            self.error_offsize = gpu.empty((batch_size,self.unitcount))
            self.bias_offsize = gpu.empty((1,batch_size))
            u.swap_pointer_and_shape(self.activation, self.activation_offsize)
            u.swap_pointer_and_shape(self.out, self.out_offsize)
            u.swap_pointer_and_shape(self.error, self.error_offsize)
            u.swap_pointer_and_shape(self.bias, self.bias_offsize)            
        elif self.activation_offsize.shape[2] != batch_size:
            print 'swap2'
            del self.activation
            del self.out
            del self.error
            del self.bias
            self.create_buffers(batch_size)
        else:
            u.swap_pointer_and_shape(self.activation, self.activation_offsize)
            u.swap_pointer_and_shape(self.out, self.out_offsize)
            u.swap_pointer_and_shape(self.error, self.error_offsize)
            u.swap_pointer_and_shape(self.bias, self.bias_offsize)    
    
    def handle_input_size(self, batch_size):
        if self.w_next==None: self.create_weights()
        if self.activation == None: self.create_buffers(batch_size)
        elif self.activation.shape[2] != batch_size: self.handle_offsize(batch_size)         
        if self.next_layer: self.next_layer.handle_input_size(batch_size)
     
    @property
    def root(self):
        root = self
        while root.next_layer: root = root.next_layer
        return root    
        
    def forward(self, data=None, target=None,inTrainingMode=True):       
        if data is not None:
            shape = u.handle_shape(data.shape)
            self.unitcount = shape[3] 
            self.handle_input_size(shape[2])           
            self.root.target = target
            self.funcs.activation(data, self.activation, self.out, inTrainingMode)
        else:
            gpu.dot(self.prev_layer.out,self.prev_layer.w_next,self.activation)             
            self.funcs.activation(self.activation, self.activation, self.out, inTrainingMode)  
            
        if self.next_layer: self.next_layer.forward(None, None, inTrainingMode)
        
    def predict(self, data):
        self.forward(data, None,False)   
        if type(self.root.funcs) == Softmax: return gpu.argmax(self.root.out)
        else: return self.root.out 
        
    def backward(self):
        self.backward_errors()
        self.backward_grads()       
        
    def backward_errors(self):
        if self.next_layer: self.next_layer.backward_errors()
        else: 
            gpu.sub(self.out,self.target,self.error)
            return
        
        
        if type(self.funcs) is Input: 
            self.backward_grads() 
            return
        
        self.funcs.grad(self.activation,self.out)
        gpu.dotT(self.next_layer.error, self.w_next, self.error)
        gpu.mul(self.error, self.out, self.error)
        
    def backward_grads(self):   
        if self.target: return
        gpu.Tdot(self.activation, self.next_layer.error, self.w_grad_next)
        if self.next_layer: self.next_layer.backward_grads()
        
    def accumulate_error(self):
        predicted_labels = gpu.argmax(self.root.out) 
        target_labels = gpu.argmax(self.root.target)
        gpu.equal(predicted_labels, target_labels, target_labels)
        error = 1.0-(target_labels.sum()/self.out.shape[2])
        self.current_error.append(error) 
        self.current_SE.append(np.array(self.current_error).std()/len(self.current_error))
        
        
    def print_reset_error(self, error_name='Train'):
        error = np.array(self.current_error).mean()
        CI_lower = error-(self.current_SE[-1]*1.96)
        CI_upper = error+(self.current_SE[-1]*1.96)
        self.error_epochs.append(error)
        self.confidence_interval_epochs.append([CI_lower, CI_upper])
        u.log_and_print('{1} error: {0}\t ({2},{3})'.format(np.round(error,4),error_name, np.round(CI_lower,4),np.round(CI_upper,4)))        
        del self.current_error
        del self.current_SE
        self.current_error = []
        self.current_SE = []
        return error
        
        
    def weight_update(self):
        if self.next_layer:         
            lib.funcs.inp_RMSProp(self.m_next.pt, self.w_grad_next.pt, ct.c_float(self.config['momentum']),ct.c_float(self.config['learning_rate']), self.out.shape[2])
            gpu.sub(self.w_next, self.w_grad_next, self.w_next)
            self.next_layer.weight_update()
        
        
    def set_config_value(self, key, value):
        self.config[key] = value
        if key == 'dropout' and self.prev_layer: self.funcs.dropout = value
        if key == 'input_dropout': 
            self.funcs.dropout = value
            return
            
        if self.next_layer:
            self.next_layer.set_config_value(key, value)
        
            
        
        
        
        
        
        


