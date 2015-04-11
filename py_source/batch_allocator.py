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


class batch_allocator(object):
    def __init__(self, data, labels, cv_percent, test_percent, batch_size):
        if len(labels.shape) ==1: labels = u.create_t_matrix(labels)
        elif labels.shape[0]==1 or labels.shape[1]==1: labels = u.create_t_matrix(labels)    
        self.size = labels.shape[0]
        self.shapes = [data.shape[1],labels.shape[1]]
        self.p_allocator = lib.funcs.fBatchAllocator()
            
        shape = u.handle_shape(data.shape)  
        shape_label = u.handle_shape(labels.shape)
        
        data = np.float32(np.asanyarray(data,order='F'))
        data = data.reshape(shape)
        labels = np.float32(np.asanyarray(labels,order='F'))
        labels = labels.reshape(shape_label)               
        
        self.batch_size = batch_size
        self.cv_percent = cv_percent
        self.test_percent = test_percent
        self.current = None
        self.next_layer = None
        self.offsize_X = []
        self.offsize_y = []
        
        self.set_type = 'train'  
        self.set_type_prev = 'train'          
        
        self.next_batch_idx = 0
        
        #self.X = array(None, lib.funcs.fto_pinned(shape[0],shape[1],shape[2],shape[3],data.ctypes.data_as(ct.POINTER(ct.c_float))))
        #self.y = array(None, lib.funcs.fto_pinned(shape_label[0],shape_label[1],shape_label[2],shape_label[3],
        #                                          labels.ctypes.data_as(ct.POINTER(ct.c_float))))
        self.X = data
        self.y = labels
        
        self.set_batch_sizes()
        self.init_buffers()
    
    def set_batch_sizes(self):
        n = np.zeros((3,))
        n[0] = np.round(self.size*(1.0-self.cv_percent-self.test_percent))
        n[1] = np.round(self.size*self.cv_percent)
        n[2] = self.size-n[0]-n[1]
        
        if self.test_percent == 0.0: n[1] = self.size-n[0]; n[2] = 1
        
        self.offbatch_rows = np.int32(n % self.batch_size)
        self.end_idx = np.ceil(np.array([n[0], n[0]+n[1], n[0]+ n[1] + n[2]]))           
        
    def deallocate_buffers(self):
        for i in range(len(self.offsize_X)):
            del self.offsize_X[i]
            del self.offsize_y[i]
        del self.current
        del self.next_layer
            
        self.current = None
        self.next_layer = None
        self.offsize_X = []
        self.offsize_y = []
        
    def init_buffers(self):
        if self.current != None: self.deallocate_buffers()
        
        self.current = gpu.empty((self.batch_size, self.shapes[0]))
        self.next_layer = gpu.empty((self.batch_size, self.shapes[0]))
        self.current_y = gpu.empty((self.batch_size, self.shapes[1]))
        self.next_y = gpu.empty((self.batch_size, self.shapes[1]))
        for value in self.offbatch_rows:
            if value > 0: 
                self.offsize_X.append(gpu.empty((value,self.shapes[0])))
                self.offsize_y.append(gpu.empty((value,self.shapes[1])))
            else: 
                self.offsize_X.append(gpu.empty((1,1,1,1)))        
                self.offsize_y.append(gpu.empty((1,1,1,1)))
        
    
        
    def train(self):     
        self.set_type = 'train'    
        return self   
    
    def cv(self): 
        self.set_type = 'cv'
        return self
    
    def test(self): 
        self.set_type = 'debug'
        return self
    
    def __iter__(self):   
        self.allocate_next_batch()
        return self
    
    def next(self):
        ret_idx = self.next_batch_idx
        self.replace_current_batch()  
        self.allocate_next_batch()    
        if self.set_type == 'train' and self.next_batch_idx == 0: raise StopIteration
        if self.set_type == 'cv' and self.next_batch_idx == self.end_idx[0]: raise StopIteration
        if self.set_type == 'debug' and self.next_batch_idx == self.end_idx[1]: raise StopIteration  
        return ret_idx
    
    def handle_copy_index(self):
        idx = self.next_batch_idx+self.batch_size
        i = 0
        if self.set_type == 'cv': i = 1
        if self.set_type == 'debug': i = 2
        if idx > self.end_idx[i]:         
            if self.next_layer.shape[2] == self.batch_size:
                u.swap_pointer_and_shape(self.next_layer, self.offsize_X[i])
                u.swap_pointer_and_shape(self.next_y, self.offsize_y[i])
            return self.end_idx[i]
        else: 
            if self.next_layer.shape[2] != self.batch_size:   
                u.swap_pointer_and_shape(self.next_layer, self.offsize_X[i])
                u.swap_pointer_and_shape(self.next_y, self.offsize_y[i])
            return idx
        
    def handle_next_batch_id(self):
        if self.set_type == self.set_type_prev == 'train':            
            if self.next_batch_idx >= self.end_idx[0]: self.next_batch_idx = 0
            
        if self.set_type == self.set_type_prev == 'cv':            
            if self.next_batch_idx >= self.end_idx[1]: self.next_batch_idx  = self.end_idx[0]
            
        if self.set_type == self.set_type_prev == 'debug':
            if self.next_batch_idx >= self.end_idx[2]: self.next_batch_idx  = self.end_idx[1]
            
        if self.set_type != self.set_type_prev:            
            i = 0
            if self.set_type_prev == 'cv': i = 1
            if self.set_type_prev == 'debug': i = 2
            if self.current.shape[2] != self.batch_size:   
                u.swap_pointer_and_shape(self.current, self.offsize_X[i])
                u.swap_pointer_and_shape(self.current_y, self.offsize_y[i])
            
            if self.set_type == 'train': self.next_batch_idx = self.next_batch_idx = 0; self.set_type_prev = 'train'
            if self.set_type == 'cv': self.next_batch_idx = self.end_idx[0]; self.set_type_prev = 'cv'
            if self.set_type == 'debug': self.next_batch_idx = self.end_idx[1]; self.set_type_prev = 'debug' 
        
    @property
    def batch(self): return self.current
    @property
    def batch_y(self): return self.current_y
    def allocate_next_batch(self):    
        self.handle_next_batch_id()    
        batch = np.float32(np.asfortranarray(self.X[:,:,self.next_batch_idx:self.handle_copy_index(),:]))
        batch_y = np.float32(np.asfortranarray(self.y[:,:,self.next_batch_idx:self.handle_copy_index(),:]))
                
        lib.funcs.fallocateNextAsync(self.p_allocator, self.next_layer.pt,batch.ctypes.data_as(ct.POINTER(ct.c_float)),self.next_y.pt,batch_y.ctypes.data_as(ct.POINTER(ct.c_float)))
        
    def replace_current_batch(self): 
        lib.funcs.freplaceCurrentBatch(self.p_allocator)
        u.swap_pointer_and_shape(self.current,self.next_layer )
        u.swap_pointer_and_shape(self.current_y,self.next_y)
        self.next_batch_idx +=self.batch_size
        
        
        
            
    