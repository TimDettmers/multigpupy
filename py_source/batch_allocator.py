'''
Created on Apr 9, 2015

@author: tim
'''
from gpupy import array
import numpy as np
import util as u
from library_interface import lib
import ctypes as ct
import gpupy as gpu
import multiprocessing as mp
import threading

class batch_allocator(object):
    def __init__(self, data, labels, cv_percent, test_percent, batch_size, mode='sequential'):
        if len(labels.shape) ==1: labels = u.create_t_matrix(labels)
        elif labels.shape[0]==1 or labels.shape[1]==1: labels = np.float32(u.create_t_matrix(labels))    
        self.size = labels.shape[0]
        self.shapes = [data.shape[1],labels.shape[1]]
            
        shape = u.handle_shape(data.shape)  
        shape_label = u.handle_shape(labels.shape)
        
        self.arrays = {}
        self.mode = mode
        
        #data = np.float32(np.asanyarray(data,order='F'))
        #data = np.float32(data.reshape(shape))
        #labels = np.float32(np.asanyarray(labels,order='F'))
        #labels = np.float32(labels.reshape(shape_label))
        
        
        
        self.pinned_buffers = {}
        
                       
        
        self.batch_size = batch_size
        self.cv_percent = cv_percent
        self.test_percent = test_percent
        self.current = None
        self.next_X = None
        self.offsize_X = []
        self.offsize_y = []
        
        self.set_type = 'train'  
        self.set_type_prev = 'train'          
        
        self.next_batch_idx = 0
        
        #self.X = array(None, lib.funcs.fto_pinned(shape[0],shape[1],shape[2],shape[3],data.ctypes.data_as(ct.POINTER(ct.c_float))))
        #self.y = array(None, lib.funcs.fto_pinned(shape_label[0],shape_label[1],shape_label[2],shape_label[3],
        #                                          labels.ctypes.data_as(ct.POINTER(ct.c_float))))
        
        self.X = lib.funcs.fto_pinned(shape[0], shape[1], shape[2], shape[3], data.ctypes.data_as(ct.POINTER(ct.c_float)))
        self.y = lib.funcs.fto_pinned(shape_label[0], shape_label[1], shape_label[2], shape_label[3], labels.ctypes.data_as(ct.POINTER(ct.c_float)))
        
        #del data
        #del labels
        
        self.set_batch_sizes()
        self.init_buffers()
        
        self.net = None
        self.sample_size = None
        self.relative_error = None
        self.start_batching = True
        self.peer_access_enabled = False
        
        self.batch_buffers = {}
    
    def set_batch_sizes(self):
        n = np.zeros((3,))
        n[0] = np.round(self.size*(1.0-self.cv_percent-self.test_percent))
        n[1] = np.round(self.size*self.cv_percent)
        n[2] = self.size-n[0]-n[1]
        
        if self.test_percent == 0.0: n[1] = self.size-n[0]; n[2] = gpu.gpu_count()
        
        self.offbatch_rows = np.int32(n % self.batch_size)
        self.end_idx = np.ceil(np.array([n[0], n[0]+n[1], n[0]+ n[1] + n[2]]))     
        self.batch_count = np.ceil(np.array([n[0],n[1],n[2]])/self.batch_size)      
        
    def deallocate_buffers(self):
        for i in range(len(self.offsize_X)):
            del self.offsize_X[i]
            del self.offsize_y[i]
        del self.current
        del self.next_X
            
        self.current = None
        self.next_X = None
        self.offsize_X = []
        self.offsize_y = []
        
    def init_buffers(self):
        if self.current != None: self.deallocate_buffers()
        split = (2 if self.mode == 'parallel' else -1)        
        self.current = gpu.empty((self.batch_size, self.shapes[0]), split)
        self.next_X = gpu.empty((self.batch_size, self.shapes[0]),split)
        self.current_y = gpu.empty((self.batch_size, self.shapes[1]),split)
        self.next_y = gpu.empty((self.batch_size, self.shapes[1]),split)
        for value in self.offbatch_rows:
            if value > 0: 
                print value
                self.offsize_X.append(gpu.empty((value,self.shapes[0]),split))
                self.offsize_y.append(gpu.empty((value,self.shapes[1]),split))
            else: 
                self.offsize_X.append(gpu.empty((1,1,1,1)))        
                self.offsize_y.append(gpu.empty((1,1,1,1)))
        
    
        
    def train(self, sample_size_or_error = None):
        if sample_size_or_error:
            if sample_size_or_error > 1: self.sample_size = sample_size_or_error
            else: self.relative_error = sample_size_or_error     
        self.set_type = 'train'     
        self.start_batching = True
        return self   
    
    def cv(self, sample_size_or_error = None): 
        if sample_size_or_error:
            if sample_size_or_error > 1: self.sample_size = sample_size_or_error
            else: self.relative_error = sample_size_or_error
        self.set_type = 'cv'
        self.start_batching = True  
        return self
    
    def test(self, sample_size_or_error = None): 
        if sample_size_or_error:
            if sample_size_or_error > 1: self.sample_size = sample_size_or_error
            else: self.relative_error = sample_size_or_error
        self.set_type = 'test'  
        self.start_batching = True
        return self
    
    def __iter__(self):   
        return self
    
    def next(self):     
        
        if self.set_type == 'train' and not self.start_batching:
            if self.next_batch_idx == 0: raise StopIteration
            if self.sample_size: 
                if self.next_batch_idx  /self.batch_size > self.sample_size:
                    self.sample_size = None                   
                    self.next_batch_idx = 0     
                    raise StopIteration
        if self.set_type == 'cv' and not self.start_batching:
            if self.next_batch_idx == self.end_idx[0]: raise StopIteration  
            if self.sample_size: 
                if (self.end_idx[0]-self.next_batch_idx  )/self.batch_size > self.sample_size: 
                    self.sample_size = None
                    self.next_batch_idx = self.end_idx[0]                        
                    raise StopIteration
        if self.set_type == 'test' and not self.start_batching:
            if self.next_batch_idx == self.end_idx[1]: raise StopIteration     
            if self.sample_size: 
                if (self.end_idx[1]-self.next_batch_idx  )/self.batch_size > self.sample_size: 
                    self.sample_size = None
                    self.next_batch_idx = self.end_idx[1]                        
                    raise StopIteration   
            
        if self.net:
            if self.relative_error:
                if len(self.net.current_error) > 2:
                    error = np.mean(np.array(self.net.current_error))
                    if error > 0:
                        if 1.96*self.net.current_SE[-1]/(error) < self.relative_error: 
                            self.relative_error = None
                            raise StopIteration
                        
        if self.start_batching:            
            self.allocate_next_batch()
            self.start_batching = False 
            
        ret_idx = self.next_batch_idx  
        self.replace_current_batch()  
        self.allocate_next_batch() 
        
        return ret_idx
    
    def handle_copy_index(self):
        idx = self.next_batch_idx+self.batch_size
        i = 0
        if self.set_type == 'cv': i = 1
        if self.set_type == 'test': i = 2
        if idx > self.end_idx[i]:         
            if self.next_X.shape[2] == self.batch_size:
                u.swap_pointer_and_shape(self.next_X, self.offsize_X[i])
                u.swap_pointer_and_shape(self.next_y, self.offsize_y[i])
            return self.end_idx[i]
        else: 
            if self.next_X.shape[2] != self.batch_size:   
                u.swap_pointer_and_shape(self.next_X, self.offsize_X[i])
                u.swap_pointer_and_shape(self.next_y, self.offsize_y[i])
            return idx
        
    def handle_next_batch_id(self):
        if self.set_type == self.set_type_prev == 'train':     
            if self.next_batch_idx >= self.end_idx[0]: self.next_batch_idx = 0
            
        if self.set_type == self.set_type_prev == 'cv':        
            if self.next_batch_idx >= self.end_idx[1]: self.next_batch_idx  = self.end_idx[0]
            
        if self.set_type == self.set_type_prev == 'test':
            if self.next_batch_idx >= self.end_idx[2]: self.next_batch_idx  = self.end_idx[1]
            
        if self.set_type != self.set_type_prev:            
            i = 0
            if self.set_type_prev == 'cv': i = 1
            if self.set_type_prev == 'test': i = 2
            if self.current.shape[2] != self.batch_size:   
                u.swap_pointer_and_shape(self.current, self.offsize_X[i])
                u.swap_pointer_and_shape(self.current_y, self.offsize_y[i])
            
            if self.set_type == 'train': self.next_batch_idx = self.next_batch_idx = 0; self.set_type_prev = 'train'
            if self.set_type == 'cv': self.next_batch_idx = self.end_idx[0]; self.set_type_prev = 'cv'
            if self.set_type == 'test': self.next_batch_idx = self.end_idx[1]; self.set_type_prev = 'test' 
        
    @property
    def batch(self):
        if self.batch_buffers: return self.batch_buffers[str(self.current.shape)+'X'] 
        return self.current
    @property
    def batch_y(self):
        if self.batch_buffers: return self.batch_buffers[str(self.current_y.shape)+'y'] 
        return self.current_y
    
    
    def get_buffers(self, flag=''):
        name = str(self.next_X.shape_tensor) + str(self.next_y.shape_tensor) + flag
        if name not in self.arrays: self.arrays[name] = [gpu.empty_like(self.next_X), gpu.empty_like(self.next_y)]
        return self.arrays[name]
    
    def allocate_next_batch(self):
        self.handle_next_batch_id()    
        self.handle_copy_index()
        #self.async_pinn_thread = threading.Thread(target=work, args=(self,), kwargs={})
        #self.async_pinn_thread.start() 
        self.allocate_next_async()
    
    def allocate_next_async(self):   
        buffers = self.get_buffers()
        split = (2 if self.mode == 'parallel' else -1)        
        lib.funcs.fallocateNextAsync(gpu.p_gpupy, 
                                     buffers[0].pt,self.X, None,
                                     buffers[1].pt,self.y, None,
                                     int(self.next_batch_idx), split)
        
    def replace_current_batch(self):         
        #self.async_pinn_thread.join()
        buffers = self.get_buffers()     
        lib.funcs.freplaceCurrentBatch(gpu.p_gpupy, self.next_X.pt, self.next_y.pt, buffers[0].pt, buffers[1].pt)
        
        u.swap_pointer_and_shape(self.current,self.next_X )
        u.swap_pointer_and_shape(self.current_y,self.next_y)
        self.next_batch_idx +=self.batch_size
                
                #gpu.print_free_memory()
                
        
        
        
            
    