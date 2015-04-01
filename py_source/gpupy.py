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

lib.fexp.restype = ct.POINTER(Tensor)
lib.inp_exp.restype = ct.c_void_p
lib.flog.restype = ct.POINTER(Tensor)
lib.inp_log.restype = ct.c_void_p
lib.fsqrt.restype = ct.POINTER(Tensor)
lib.inp_sqrt.restype = ct.c_void_p
lib.flogistic.restype = ct.POINTER(Tensor)
lib.inp_logistic.restype = ct.c_void_p
lib.flogisticGrad.restype = ct.POINTER(Tensor)
lib.inp_logisticGrad.restype = ct.c_void_p
lib.ffabs.restype = ct.POINTER(Tensor)
lib.inp_abs.restype = ct.c_void_p
lib.fsquare.restype = ct.POINTER(Tensor)
lib.inp_square.restype = ct.c_void_p
lib.ffpow.restype = ct.POINTER(Tensor)
lib.inp_pow.restype = ct.c_void_p

lib.faddVectorToTensor.restype = ct.POINTER(Tensor)
lib.inp_addVectorToTensor.restype = ct.c_void_p
lib.fsubVectorToTensor.restype = ct.POINTER(Tensor)
lib.inp_subVectorToTensor.restype = ct.c_void_p
lib.fmulVectorToTensor.restype = ct.POINTER(Tensor)
lib.inp_mulVectorToTensor.restype = ct.c_void_p
lib.fdivVectorToTensor.restype = ct.POINTER(Tensor)
lib.inp_divVectorToTensor.restype = ct.c_void_p

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
        
        self.npArray = data
        
        return data        

    @property
    def T(self): return array(None, lib.fT(self.pt))         
    def __del__(self): lib.ffree(self.pt)
    def __add__(self, other): return exec_scalar_or_matrix_op(self,other, add, addScalar)
    def __sub__(self, other): return exec_scalar_or_matrix_op(self,other, sub, subScalar)
    def __mul__(self, other): return exec_scalar_or_matrix_op(self,other, mul, mulScalar)
    def __div__(self, other): return exec_scalar_or_matrix_op(self,other, div, divScalar)
    #def abs(self): return absolute(self, out=None)
    
    
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
    
def exec_scalar_or_matrix_op(x1, x2, func_matrix, func_scalar, inplace=False): 
    is_scalar =  isinstance(x2, int) or isinstance(x2, float)
    if is_scalar: 
        if inplace: func_scalar(x1,x2,x1)
        else: return func_scalar(x1,x2)
    else: 
        if inplace:func_matrix(x1,x2,x1) 
        else: return func_matrix(x1,x2)

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

def add(x1,x2,out=None):
    if out: lib.inp_add(x1.pt,x2.pt,out.pt);
    else: return array(None, lib.fadd(x1.pt,x2.pt))
    
def sub(x1,x2,out=None):
    if out: lib.inp_sub(x1.pt,x2.pt,out.pt);
    else: return array(None, lib.fsub(x1.pt,x2.pt))
    
def mul(x1,x2,out=None):
    if out: lib.inp_mul(x1.pt,x2.pt,out.pt);
    else: return array(None, lib.fmul(x1.pt,x2.pt))
    
def div(x1,x2,out=None):
    if out: lib.inp_div(x1.pt,x2.pt,out.pt);
    else: return array(None, lib.fdiv(x1.pt,x2.pt))        

def addScalar(x1,flt,out=None):
    if out: lib.inp_scalarAdd(x1.pt, ct.c_float(float(flt)), out.pt)
    else: return array(None, lib.fscalarAdd(x1.pt, ct.c_float(float(flt))))
    
def subScalar(x1,flt,out=None):
    if out: lib.inp_scalarAdd(x1.pt, ct.c_float(-flt), out.pt)
    else: return array(None, lib.fscalarAdd(x1.pt, ct.c_float(-flt)))
    
def mulScalar(x1,flt,out=None):
    if out: lib.inp_scalarMul(x1.pt, ct.c_float(flt), out.pt)
    else: return array(None, lib.fscalarMul(x1.pt, ct.c_float(flt)))
    
def divScalar(x1,flt,out=None):
    if out: lib.inp_scalarMul(x1.pt, ct.c_float(1.0/flt), out.pt)
    else: return array(None, lib.fscalarMul(x1.pt, ct.c_float(1.0/flt)))
    
def exp(x1,out=None):
    if out: lib.inp_exp(x1.pt,out.pt);
    else: return array(None, lib.fexp(x1.pt))
    
def log(x1,out=None):
    if out: lib.inp_log(x1.pt,out.pt);
    else: return array(None, lib.flog(x1.pt))
    
def sqrt(x1,out=None):
    if out: lib.inp_sqrt(x1.pt,out.pt);
    else: return array(None, lib.fsqrt(x1.pt))
    
def logistic(x1,out=None):
    if out: lib.inp_logistic(x1.pt,out.pt);
    else: return array(None, lib.flogistic(x1.pt))

def logisticGrad(x1,out=None):    
    """Computes x1*(1-x1).
    :x1: Logistic Input.
    :out: Write output to this Tensor.
    """
    if out: lib.inp_logisticGrad(x1.pt,out.pt);
    else: return array(None, lib.flogisticGrad(x1.pt))

def abs(x1,out=None):
    if out: lib.inp_abs(x1.pt,out.pt);
    else: return array(None, lib.ffabs(x1.pt))
    
def square(x1,out=None):
    if out: lib.inp_square(x1.pt,out.pt);
    else: return array(None, lib.fsquare(x1.pt))
    
def power(x1,power, out=None):
    if out: lib.inp_pow(x1.pt,ct.c_float(power),out.pt);
    else: return array(None, lib.ffpow(x1.pt, ct.c_float(power)))
    
def addVectorToTensor(x1,v1, out=None):
    if out: lib.inp_addVectorToTensor(x1.pt, v1.pt, out.pt)
    else: return array(None, lib.faddVectorToTensor(x1.pt, v1.pt))
    
def subVectorToTensor(x1,v1, out=None):
    if out: lib.inp_subVectorToTensor(x1.pt, v1.pt, out.pt)
    else: return array(None, lib.fsubVectorToTensor(x1.pt, v1.pt))
    
def mulVectorToTensor(x1,v1, out=None):
    if out: lib.inp_mulVectorToTensor(x1.pt, v1.pt, out.pt)
    else: return array(None, lib.fmulVectorToTensor(x1.pt, v1.pt))
    
def divVectorToTensor(x1,v1, out=None):
    if out: lib.inp_divVectorToTensor(x1.pt, v1.pt, out.pt)
    else: return array(None, lib.fdivVectorToTensor(x1.pt, v1.pt))
    
