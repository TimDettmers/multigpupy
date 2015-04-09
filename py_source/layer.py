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
        else: gpu.copy(my_activation, my_output)
        
    def grad(self, x1, out): return self.gpu_func_grad(x1, out)  
    
class Logistic(ActivationFunc): 
    def __init__(self):  
        super(Logistic, self).__init__(0.5, gpu.logistic, gpu.logistic_grad)
            
class ReLU(ActivationFunc):      
    def __init__(self):  
        super(ReLU, self).__init__(0.5, gpu.ReLU, gpu.ReLU_grad)      
        
class Input(ActivationFunc): 
    def __init__(self):  
        super(Input, self).__init__(0.2, gpu.linear, gpu.linear)
    
class Softmax(ActivationFunc):
    def __init__(self):   
        super(Softmax, self).__init__(0.0, gpu.softmax, gpu.linear)

class Layer(object):
    def __init__(self, unitcount=0, activation_function=Input()):
        self.p_layer = lib.funcs.fLayer()
        self.inTrainingMode = True
        self.w_next = None
        self.activation = None
        self.activation_offsize = None         
        self.funcs = activation_function
        self.unitcount = unitcount
        self.next = None
        self.prev = None
        self.id = 0
        
    def add(self,next_layer):
        if self.next:            
            self.next.add(next_layer)
            return
        
        if type(next_layer) is Layer:            
            self.next = next_layer
            next_layer.prev = self   
            next_layer.id = self.id +1
        else:
            self.funcs = next_layer
    
    def create_weights(self):
        if self.next:
            self.w_next = gpu.array(np.float32(np.random.normal(0,0.1,(self.unitcount,self.next.unitcount))))
            self.m_next = gpu.zeros((self.unitcount, self.next.unitcount))
            self.w_grad_next = gpu.zeros((self.unitcount, self.next.unitcount))
            if self.next: self.next.create_weights()
        
    def create_buffers(self, batch_size):
        print 'buffers for id ', self.id
        self.activation = gpu.empty((batch_size,self.unitcount))
        self.out = gpu.empty((batch_size,self.unitcount))
        self.error = gpu.empty((batch_size,self.unitcount))
        self.bias = gpu.empty((1,batch_size))
        
    def handle_offsize(self, batch_size):
        print 'handle offsize'
        if self.activation_offsize == None:
            self.activation_offsize = gpu.empty((batch_size,self.unitcount))
            self.out_offsize = gpu.empty((batch_size,self.unitcount))
            self.error_offsize = gpu.empty((batch_size,self.unitcount))
            self.bias_offsize = gpu.empty((1,batch_size))
        elif self.activation_offsize.shape[0] != batch_size:
            del self.activation
            del self.out
            del self.error
            del self.bias
            self.create_buffers(batch_size)
        else:
            self.swap_pointer(self.activation, self.activation_offsize)
            self.swap_pointer(self.out, self.out_offsize)
            self.swap_pointer(self.error, self.error_offsize)
            self.swap_pointer(self.bias, self.bias_offsize)     
    
    def handle_input_size(self, batch_size):
        if self.w_next==None: self.create_weights()
        if self.activation == None: self.create_buffers(batch_size)
        elif self.activation.shape[0] != batch_size: self.handle_offsize(batch_size)
        if self.next: self.next.handle_input_size(batch_size)
        
        
    def forward(self, data=None):        
        if data: 
            self.unitcount = data.shape[1] 
            self.handle_input_size(data.shape[0])          
        if data is not None: self.funcs.activation(data, self.activation, self.out, self.inTrainingMode)
        else:
            gpu.dot(self.prev.out,self.prev.w_next,self.activation)             
            self.funcs.activation(self.activation, self.activation, self.out, self.inTrainingMode)  
        if self.next: self.next.forward()
        
        
        
        


