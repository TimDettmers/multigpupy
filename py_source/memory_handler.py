'''
Created on Apr 30, 2015

@author: tim
'''
import gpupy as gpu
import numpy as np

class MemoryHandler(object):
    def __init__(self):
        self.arrays = {}
        self.sync_arrays = {}  
        self.compression_arrays = {}      
        
        self.usage_stats = {}  
        pass
    
    def get_arrays_for_sync(self, A, dtype=np.float32):        
        if A.id not in self.sync_arrays: 
            self.sync_arrays[A.id] = self.arrays_like(A, gpu.gpu_count(), dtype)
            self.usage_stats[A.id] = 1    
        else:    
            self.usage_stats[A.id] += 1         
        return self.sync_arrays[A.id]        
    
    def arrays_like(self, A, array_count, dtype=np.float32):
        arrays = []
        for i in range(array_count): 
            if dtype == np.float32: arrays.append(gpu.zeros_like(A))
            elif dtype == np.float16: arrays.append(gpu.empty_ushort_like(A))
            elif dtype == np.uint32: arrays.append(gpu.empty_uint_like(A))
            elif dtype == np.char: arrays.append(gpu.empty_char_like(A))
            else: raise ValueError('Type not supported', dtype)
            
        return arrays
    
    def array_list_to_id_list(self, arrays):
        return [array.id for array in arrays]
    
    
                
    
    
    
