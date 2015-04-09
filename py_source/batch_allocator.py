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

p_allocator = lib.funcs.fBatchAllocator()

class batch_allocator(object):
    def __init__(self, data, labels, cv_percent, test_percent, batch_size):     
        print labels.shape  
        if len(labels.shape) ==1: labels = u.create_t_matrix(labels)
        elif labels.shape[0]==1 or labels.shape[1]==1: labels = u.create_t_matrix(labels)    
        self.size = labels.shape[0]
        self.shapes = [data.shape[1],labels.shape[1]]
            
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
        self.next = None
        self.offsize = []
        self.offsize_y = []            
        
        self.next_batch_id = 0
        
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
        
        self.sizes = n
        self.batches =  np.int32(np.ceil(n/self.batch_size))
        self.offbatch_rows = np.int32(n-np.floor((n/self.batch_size)))
        
    def deallocate_buffers(self):
        for i in range(len(self.offsize)):
            del self.offsize[i]
            del self.offsize_y[i]
        del self.current
        del self.next
            
        self.current = None
        self.next = None
        self.offsize = []
        self.offsize_y = []
        
    def init_buffers(self):
        if self.current != None: self.deallocate_buffers()
        
        self.current = gpu.empty((self.batch_size, self.shapes[0]))
        self.next = gpu.empty((self.batch_size, self.shapes[0]))
        self.current_y = gpu.empty((self.batch_size, self.shapes[1]))
        self.next_y = gpu.empty((self.batch_size, self.shapes[1]))
        for value in self.offbatch_rows:
            if value > 0: 
                self.offsize.append(gpu.empty((value,self.shapes[0])))
                self.offsize_y.append(gpu.empty((value,self.shapes[1])))
            else: 
                self.offsize.append(gpu.empty((1,1,1,1)))        
                self.offsize_y.append(gpu.empty((1,1,1,1)))
        
    
        
    @property
    def batch(self): return self.current
    @property
    def batch_y(self): return self.current_y
    def allocate_next_batch(self):
        if self.next_batch_id >= self.batches[0]: self.next_batch_id = 0        
        batch = np.float32(np.asfortranarray(self.X[:,:,self.next_batch_id*self.batch_size:(self.next_batch_id+1)*self.batch_size,:]))
        batch_y = np.float32(np.asfortranarray(self.y[:,:,self.next_batch_id*self.batch_size:(self.next_batch_id+1)*self.batch_size,:]))
        lib.funcs.fallocateNextAsync(p_allocator, self.next.pt,batch.ctypes.data_as(ct.POINTER(ct.c_float)),self.next_y.pt,batch_y.ctypes.data_as(ct.POINTER(ct.c_float)))
        
    def replace_current_batch(self): 
        lib.funcs.freplaceCurrentBatch(p_allocator)
        u.swap_pointer(self.current, self.next)
        u.swap_pointer(self.current_y, self.next_y)
        self.next_batch_id +=1
        
        
        
            
    