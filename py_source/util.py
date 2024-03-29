'''
Created on Mar 24, 2015

@author: tim
'''
import numpy as np
import inspect
import logging


def handle_shape(shape):    
    assert len(shape)<=4, "GPUpy only supports up tp four dimensions!"
    if len(shape) == 1: shape = (1,1,1) + shape
    if len(shape) == 2: shape = (1,1) + shape
    if len(shape) == 3: shape = (1,) + shape
    return shape 
    
def handle_dim(d0, d1, d2, d3):
    if d1 == None: d3 = d0; d2=1; d1=1; d0=1
    if d2 == None: d3 = d1; d2=d0; d1=1; d0=1
    if d3 == None: d3 = d2; d2=d1; d1=d0; d0=1
    return d0,d1,d2,d3

def convert_shape_value(value, dim, start=True):
    if value == None and start: return 0
    elif value == None and not start: return dim
    elif value < 0: return dim+value
    elif value > dim: return dim
    else: return value

def handle_selectors(selectors, shape):
    if type(selectors) == type(slice(1)): selectors = [selectors]
    select = [[0,shape[0]], [0,shape[1]],[0,shape[2]],[0,shape[3]]]
    select_buffer = [[0,shape[0]], [0,shape[1]],[0,shape[2]],[0,shape[3]]]
    for i, selector in enumerate(selectors):
        dimGreaterOneCount = 0
        for idx, (start, stop) in enumerate(select):
            if stop > 1:
                if i ==  dimGreaterOneCount:
                    select_buffer[idx][0] = convert_shape_value(selector.start, stop)
                    select_buffer[idx][1] = convert_shape_value(selector.stop, stop,False)                    
                dimGreaterOneCount+=1
               
    return select_buffer
    
            
def create_t_matrix(y, classes = None):
    if not classes: classes = np.max(y)+1    
    t = np.zeros((y.shape[0], classes), dtype=np.float32)
    for i in range(y.shape[0]):
        t[np.int32(i), np.int32(y[i])] = 1.0
        
    return t

def softmax(X):
    '''numerically stable softmax function
    '''
    max_row_values = np.matrix(np.max(X,axis=1)).T
    result = np.exp(X - max_row_values)
    sums = np.matrix(np.sum(result,axis=1))        
    return result/sums      
            
def swap_pointer(x1, x2):
    swap = x1.pt    
    x1.pt = x2.pt
    x2.pt = swap  
    
def swap_pointer_and_shape(x1, x2):
    swap = x1.pt    
    x1.pt = x2.pt
    x2.pt = swap
    swap = x1.shape
    x1.shape = x2.shape
    x2.shape = swap   
    swap = x1.shape_tensor
    x1.shape_tensor = x2.shape_tensor
    x2.shape_tensor = swap   

def create_uniform_rdm_weight(input_size,output_size):
    rdm = np.random.RandomState(1234)        
    return np.float32(rdm.uniform(low=-4*np.sqrt(6./(input_size+output_size)),
                    high=4*np.sqrt(6./(input_size+output_size)),
                    size=(input_size,output_size)))
    

def get_current_function_name():
    return inspect.stack()[1][3]

def log_and_print(message):
    logging.info(message)
    print message

