'''
Created on Mar 21, 2015

@author: tim
'''

import numpy as np
import ctypes as ct
import util as u
lib = ct.cdll.LoadLibrary('./gpupylib.so')



class Tensor(ct.Structure):
    _fields_ = [('batches', ct.c_int),
                ('maps', ct.c_int),
                ('rows', ct.c_int),
                ('cols', ct.c_int),
                ('bytes', ct.c_size_t),
                ('size', ct.c_int),
                ('data', ct.POINTER(ct.c_float))]
    
    def __init__(self): pass

lib.fempty.restype = ct.POINTER(Tensor)
lib.fzeros.restype = ct.POINTER(Tensor)
lib.fones.restype = ct.POINTER(Tensor)
lib.ftocpu.restype = ct.POINTER(Tensor)
lib.ftogpu.restype = ct.c_void_p
lib.fadd.restype = ct.POINTER(Tensor)
lib.finplaceAdd.restype = ct.c_void_p

def __init__(): pass

class array(object):
    def __init__(self, npArray = None, mat_pointer = None):
        self.shape = None
                     
        if npArray != None:
            npArray = np.float32(npArray)
            mat_pointer = empty(npArray.shape).pt
            lib.ftogpu(mat_pointer, npArray.ctypes.data_as(ct.POINTER(ct.c_float)))
            self.shape = npArray.shape 
            
        if mat_pointer:             
            self.shape_tensor = (mat_pointer.contents.batches, mat_pointer.contents.maps,
                                 mat_pointer.contents.rows, mat_pointer.contents.cols)
            if not self.shape: self.shape = self.shape_tensor
            
               
        self.pt = mat_pointer
        self.npArray = npArray      
        pass
    
    def tocpu(self):
        data = np.empty(self.shape, dtype=np.float32)
        lib.ftocpu(self.pt, data.ctypes.data_as(ct.POINTER(ct.c_float)))        
        self.npArray = data    
        
        if data.shape[0] == 1 and data.shape[1] == 1: data = data.reshape(data.shape[2], data.shape[3])
        
        self.npArray = data
        
        return data



def zeros(shape):
    shape = u.handle_shape(shape)
    out = array(None, lib.fzeros(shape[0],shape[1],shape[2],shape[3]))
    return out


def ones(shape):
    shape = u.handle_shape(shape)
    out = array(None, lib.fones(shape[0],shape[1],shape[2],shape[3]))
    return out

def empty(shape):
    shape = u.handle_shape(shape)
    out = array(None, lib.fempty(shape[0],shape[1],shape[2],shape[3]))
    return out

def add(A,B,out=None):
    if out:
        lib.finplaceAdd(A.pt,B.pt,out.pt);
        pass
    else:
        return array(None, lib.fadd(A.pt,B.pt))