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
lib.ffree.restype = ct.c_void_p
lib.fT.restype = ct.POINTER(Tensor)
lib.inp_T.restype = ct.c_void_p

lib.fadd.restype = ct.POINTER(Tensor)
lib.inp_add.restype = ct.c_void_p
lib.fsub.restype = ct.POINTER(Tensor)
lib.inp_sub.restype = ct.c_void_p
lib.fmul.restype = ct.POINTER(Tensor)
lib.inp_mul.restype = ct.c_void_p
lib.fdiv.restype = ct.POINTER(Tensor)
lib.inp_div.restype = ct.c_void_p
lib.fscalarAdd.restype = ct.POINTER(Tensor)
lib.inp_scalarAdd.restype = ct.c_void_p
lib.fscalarMul.restype = ct.POINTER(Tensor)
lib.inp_scalarMul.restype = ct.c_void_p

def __init__(): pass

class array(object):
    def __init__(self, npArray = None, mat_pointer = None):
        self.shape = None
        self.dummy = False
                     
        if type(npArray) == type(np.array(1)):
            npArray = np.float32(npArray)
            shape = u.handle_shape(npArray.shape)
            mat_pointer = lib.fempty(shape[0],shape[1],shape[2],shape[3])
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
        if data.shape[0] == 1 and data.shape[1] > 1: data = data.reshape(data.shape[1], data.shape[2], data.shape[3])
        
        print data.shape[0], data.shape[1]
        print data.shape
        
        self.npArray = data
        
        return data
    

        
        

    @property
    def T(self): return array(None, lib.fT(self.pt))         
    def __del__(self): lib.ffree(self.pt)
    def __add__(self, other): return exec_scalar_or_matrix_op(self,other, add, addScalar)
    def __sub__(self, other): return exec_scalar_or_matrix_op(self,other, sub, subScalar)
    def __mul__(self, other): return exec_scalar_or_matrix_op(self,other, mul, mulScalar)
    def __div__(self, other): return exec_scalar_or_matrix_op(self,other, div, divScalar)
    
    
    def __iadd__(self, other): 
        exec_scalar_or_matrix_op(self, other, add, addScalar, inplace=True)
        return self
    
    def __isub__(self, other): 
        exec_scalar_or_matrix_op(self, other, sub, subScalar, inplace=True)
        return self
    
    def __imul__(self, other): 
        exec_scalar_or_matrix_op(self, other, mul, mulScalar, inplace=True)
        return self
    
    def __idiv__(self, other): 
        exec_scalar_or_matrix_op(self, other, div, divScalar, inplace=True)
        return self
    
def exec_scalar_or_matrix_op(A, B, func_matrix, func_scalar, inplace=False): 
    is_scalar =  isinstance(B, int) or isinstance(B, float)
    if is_scalar: 
        if inplace: func_scalar(A,B,A)
        else: return func_scalar(A,B)
    else: 
        if inplace:func_matrix(A,B,A) 
        else: return func_matrix(A,B)

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
    if out: lib.inp_add(A.pt,B.pt,out.pt);
    else: return array(None, lib.fadd(A.pt,B.pt))
    
def sub(A,B,out=None):
    if out: lib.inp_sub(A.pt,B.pt,out.pt);
    else: return array(None, lib.fsub(A.pt,B.pt))
    
def mul(A,B,out=None):
    if out: lib.inp_mul(A.pt,B.pt,out.pt);
    else: return array(None, lib.fmul(A.pt,B.pt))
    
def div(A,B,out=None):
    if out: lib.inp_div(A.pt,B.pt,out.pt);
    else: return array(None, lib.fdiv(A.pt,B.pt))        

def addScalar(A,flt,out=None):
    if out: lib.inp_scalarAdd(A.pt, ct.c_float(float(flt)), out.pt)
    else: return array(None, lib.fscalarAdd(A.pt, ct.c_float(float(flt))))
    
def subScalar(A,flt,out=None):
    if out: lib.inp_scalarAdd(A.pt, ct.c_float(-flt), out.pt)
    else: return array(None, lib.fscalarAdd(A.pt, ct.c_float(-flt)))
    
def mulScalar(A,flt,out=None):
    if out: lib.inp_scalarMul(A.pt, ct.c_float(flt), out.pt)
    else: return array(None, lib.fscalarMul(A.pt, ct.c_float(flt)))
    
def divScalar(A,flt,out=None):
    if out: lib.inp_scalarMul(A.pt, ct.c_float(1.0/flt), out.pt)
    else: return array(None, lib.fscalarMul(A.pt, ct.c_float(1.0/flt)))
    
