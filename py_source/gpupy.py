'''
Created on Mar 21, 2015

@author: tim
'''

import numpy as np
import ctypes as ct
import util as u
from library_interface import lib

p_gpupy = lib.funcs.fGPUpy()

class Slice():
    def __init__(self, slice_pointer):
        self.pt = slice_pointer
        self.batch = slice(int(slice_pointer.contents.batch_start), int(slice_pointer.contents.batch_stop), None)
        self.map = slice(int(slice_pointer.contents.map_start), int(slice_pointer.contents.map_stop), None) 
        self.row = slice(int(slice_pointer.contents.row_start), int(slice_pointer.contents.row_stop), None) 
        self.col = slice(int(slice_pointer.contents.col_start), int(slice_pointer.contents.col_stop), None) 
        
    def setSliceValues(self, selectors):
        selectors = u.handle_selectors(selectors)
        self.batch = selectors[0]
        self.map = selectors[1]
        self.row = selectors[2]
        self.col = selectors[3]
        
        self.pt.contents.batch_start = self.batch.start
        self.pt.contents.batch_stop = self.batch.stop
        self.pt.contents.map_start = self.map.start
        self.pt.contents.map_stop = self.map.stop
        self.pt.contents.row_start = self.row.start
        self.pt.contents.row_stop = self.row.stop
        self.pt.contents.col_start = self.col.start
        self.pt.contents.col_stop = self.col.stop
                     

class array(object):
    def __init__(self, npArray = None, mat_pointer = None):
        self.shape = None
        self.dummy = False
                     
        if type(npArray) == type(np.array(1)):
            npArray = np.float32(npArray)
            shape = u.handle_shape(npArray.shape)
            mat_pointer = lib.funcs.fempty(shape[0],shape[1],shape[2],shape[3])
            lib.funcs.ftogpu(mat_pointer, npArray.ctypes.data_as(ct.POINTER(ct.c_float)))
            self.shape = npArray.shape 
            
        if mat_pointer:             
            self.shape_tensor = (mat_pointer.contents.batches, mat_pointer.contents.maps,
                                 mat_pointer.contents.rows, mat_pointer.contents.cols)
            if not self.shape: self.shape = self.shape_tensor
            
               
        self.pt = mat_pointer
        self.npArray = npArray      
        pass
    
    def __getitem__(self, selectors):
        select_buffer = u.handle_selectors(selectors, self.shape_tensor)      
        
        S = emptySlice()        
        S.pt.contents.batch_start = ct.c_int32(select_buffer[0][0])
        S.pt.contents.batch_stop = ct.c_int32(select_buffer[0][1])
        S.pt.contents.map_start = select_buffer[1][0]
        S.pt.contents.map_stop = select_buffer[1][1]
        S.pt.contents.row_start = select_buffer[2][0]
        S.pt.contents.row_stop = select_buffer[2][1]
        S.pt.contents.col_start = select_buffer[3][0]
        S.pt.contents.col_stop = select_buffer[3][1]     
        
        #print S.pt.contents.batch_start, S.pt.contents.batch_stop  
        
        return array(None, lib.funcs.fslice(self.pt,S.pt))
        pass
    
    
    def tocpu(self):
        data = np.empty(self.shape, dtype=np.float32)
        lib.funcs.ftocpu(self.pt, data.ctypes.data_as(ct.POINTER(ct.c_float)))        
        self.npArray = data
        
        if data.shape[0] == 1 and data.shape[1] == 1 and data.shape[2] == 1: data = data.reshape(data.shape[3])
        if data.shape[0] == 1 and data.shape[1] == 1: data = data.reshape(data.shape[2], data.shape[3])
        if data.shape[0] == 1 and data.shape[1] > 1: data = data.reshape(data.shape[1], data.shape[2], data.shape[3])
        
        self.npArray = data
        
        return data        

    @property
    def T(self): return array(None, lib.funcs.fT(self.pt))   
      
    def __del__(self): lib.funcs.ffree(self.pt)
    def __add__(self, other): return apply_func(self,other, lib.funcs.fadd, lib.funcs.fscalarAdd, lib.funcs.faddVectorToTensor)
    def __sub__(self, other): return apply_func(self,other, lib.funcs.fsub, lib.funcs.fscalarSub, lib.funcs.fsubVectorToTensor)
    def __mul__(self, other): return apply_func(self,other, lib.funcs.fmul, lib.funcs.fscalarMul, lib.funcs.fmulVectorToTensor)
    def __div__(self, other): return apply_func(self,other, lib.funcs.fdiv, lib.funcs.fscalarDiv, lib.funcs.fdivVectorToTensor)
    def __eq__(self, other): return equal(self,other)
    def __lt__(self, other):  return less(self,other)
    def __gt__(self, other): return greater(self,other)
    def __ge__(self, other): return greater_equal(self,other)
    def __le__(self, other): return less_equal(self,other)
    def __ne__(self, other): return not_equal(self,other)
    #def abs(self): return absolute(self, out=None)
    
    
    def __iadd__(self, other): 
        apply_func(self, other, lib.funcs.inp_add, lib.funcs.inp_scalarAdd, lib.funcs.inp_addVectorToTensor, out=self)
        return self
    
    def __isub__(self, other): 
        apply_func(self, other, lib.funcs.inp_sub, lib.funcs.inp_scalarSub, lib.funcs.inp_subVectorToTensor, out=self)
        return self
    
    def __imul__(self, other): 
        apply_func(self, other, lib.funcs.inp_mul, lib.funcs.inp_scalarMul, lib.funcs.inp_mulVectorToTensor, out=self)
        return self
    
    def __idiv__(self, other): 
        apply_func(self, other, lib.funcs.inp_div, lib.funcs.inp_scalarDiv, lib.funcs.inp_divVectorToTensor, out=self)
        return self
    
def apply_func(x1, x2, func_matrix, func_scalar, func_vector = None, out=None): 
    is_scalar =  isinstance(x2, int) or isinstance(x2, float)
    if not is_scalar: 
        is_vectors = [is_vector(x1), is_vector(x2)]   
    if is_scalar: 
        if out: func_scalar(x1.pt,ct.c_float(x2),out.pt)
        else: return array(None,func_scalar(x1.pt,ct.c_float(x2)))
    elif any(is_vectors): 
        if is_vectors[0] and out: func_vector(x2.pt, x1.pt,out.pt)
        elif is_vectors[0]: return array(None,func_vector(x2.pt, x1.pt))
        elif is_vectors[1] and out: func_vector(x1.pt, x2.pt,out.pt)
        elif is_vectors[1]: return array(None,func_vector(x1.pt, x2.pt))
    else:
        if out: func_matrix(x1.pt,x2.pt,out.pt)
        else: return array(None,func_matrix(x1.pt,x2.pt))
        
def gpu_count(): return lib.funcs.fGPUCount(p_gpupy)  
        
def is_vector(x):
    if type(x) != type(array): False
    not_one_count = 0    
    for dim in x.shape:
        if dim > 1: not_one_count+=1
    return (False if not_one_count > 1 else True)

def zeros(shape):
    shape = u.handle_shape(shape)
    out = array(None, lib.funcs.fzeros(shape[0],shape[1],shape[2],shape[3]))
    return out


def ones(shape):
    shape = u.handle_shape(shape)
    out = array(None, lib.funcs.fones(shape[0],shape[1],shape[2],shape[3]))
    return out

def empty(shape):
    shape = u.handle_shape(shape)
    out = array(None, lib.funcs.fempty(shape[0],shape[1],ct.c_int32(shape[2]),shape[3]))
    return out

def add(x1,x2,out=None):
    if out: apply_func(x1,x2, lib.funcs.inp_add, lib.funcs.inp_scalarAdd, lib.funcs.inp_addVectorToTensor, out)
    else: return apply_func(x1,x2, lib.funcs.fadd, lib.funcs.fscalarAdd, lib.funcs.faddVectorToTensor)
    
def sub(x1,x2,out=None):
    if out: apply_func(x1,x2, lib.funcs.inp_sub, lib.funcs.inp_scalarSub, lib.funcs.inp_subVectorToTensor, out)
    else: return apply_func(x1,x2, lib.funcs.fsub, lib.funcs.fscalarSub, lib.funcs.fsubVectorToTensor)
    
def mul(x1,x2,out=None):
    if out: apply_func(x1,x2, lib.funcs.inp_mul, lib.funcs.inp_scalarMul, lib.funcs.inp_mulVectorToTensor, out)
    else: return apply_func(x1,x2, lib.funcs.fmul, lib.funcs.fscalarMul, lib.funcs.fmulVectorToTensor)
    
def div(x1,x2,out=None):
    if out: apply_func(x1,x2, lib.funcs.inp_div, lib.funcs.inp_scalarDiv, lib.funcs.inp_divVectorToTensor, out)
    else: return apply_func(x1,x2, lib.funcs.fdiv, lib.funcs.fscalarDiv, lib.funcs.fdivVectorToTensor)       
    
def exp(x1,out=None):
    if out: lib.funcs.inp_exp(x1.pt,out.pt);
    else: return array(None, lib.funcs.fexp(x1.pt))
    
def log(x1,out=None):
    if out: lib.funcs.inp_log(x1.pt,out.pt);
    else: return array(None, lib.funcs.flog(x1.pt))
    
def sqrt(x1,out=None):
    if out: lib.funcs.inp_pow(x1.pt,ct.c_float(0.5),out.pt);
    else: return array(None, lib.funcs.ffpow(x1.pt, ct.c_float(0.5)))    
    
def logistic(x1,out=None):
    if out: lib.funcs.inp_logistic(x1.pt,out.pt);
    else: return array(None, lib.funcs.flogistic(x1.pt))

def logisticGrad(x1,out=None):    
    """Computes x1*(1-x1).
    :x1: Logistic Input.
    :out: Write output to this Tensor.
    """
    if out: lib.funcs.inp_logisticGrad(x1.pt,out.pt);
    else: return array(None, lib.funcs.flogisticGrad(x1.pt))

def abs(x1,out=None):    
    if out: lib.funcs.inp_abs(x1.pt,out.pt);
    else: return array(None, lib.funcs.ffabs(x1.pt))
    
def square(x1,out=None):
    if out: lib.funcs.inp_pow(x1.pt,ct.c_float(2.0),out.pt);
    else: return array(None, lib.funcs.ffpow(x1.pt, ct.c_float(2.0)))    
    
def power(x1,power, out=None):
    if out: lib.funcs.inp_pow(x1.pt,ct.c_float(power),out.pt);
    else: return array(None, lib.funcs.ffpow(x1.pt, ct.c_float(power)))    
    
def equal(x1,x2,out=None):
    if out: apply_func(x1,x2, lib.funcs.inp_eq, lib.funcs.inp_scalar_eq, lib.funcs.inp_vec_eq, out)
    else: return apply_func(x1,x2, lib.funcs.feq, lib.funcs.fscalar_eq, lib.funcs.fvec_eq)
    
def less(x1,x2,out=None):
    if out: apply_func(x1,x2, lib.funcs.inp_lt, lib.funcs.inp_scalar_lt, lib.funcs.inp_vec_lt, out)
    else: return apply_func(x1,x2, lib.funcs.flt, lib.funcs.fscalar_lt, lib.funcs.fvec_lt)
        
def less_equal(x1,x2,out=None):
    if out: apply_func(x1,x2, lib.funcs.inp_le, lib.funcs.inp_scalar_le, lib.funcs.inp_vec_le, out)
    else: return apply_func(x1,x2, lib.funcs.fle, lib.funcs.fscalar_le, lib.funcs.fvec_le)
        
def greater(x1,x2,out=None):
    if out: apply_func(x1,x2, lib.funcs.inp_gt, lib.funcs.inp_scalar_gt, lib.funcs.inp_vec_gt, out)
    else: return apply_func(x1,x2, lib.funcs.fgt, lib.funcs.fscalar_gt, lib.funcs.fvec_gt)
    
def greater_equal(x1,x2,out=None):
    if out: apply_func(x1,x2, lib.funcs.inp_ge, lib.funcs.inp_scalar_ge, lib.funcs.inp_vec_ge, out)
    else: return apply_func(x1,x2, lib.funcs.fge, lib.funcs.fscalar_ge, lib.funcs.fvec_ge)
  
def not_equal(x1,x2,out=None):
    if out: apply_func(x1,x2, lib.funcs.inp_ne, lib.funcs.inp_scalar_ne, lib.funcs.inp_vec_ne, out)
    else: return apply_func(x1,x2, lib.funcs.fne, lib.funcs.fscalar_ne, lib.funcs.fvec_ne)  
    
def emptySlice():
    return Slice(lib.funcs.femptySlice())

def dot(a,b,out=None):
    if out: lib.funcs.inp_dot(p_gpupy, a.pt, b.pt, out.pt)
    else: return array(None, lib.funcs.fdot(p_gpupy, a.pt,b.pt))
    
def Tdot(a,b,out=None):
    if out: lib.funcs.inp_Tdot(p_gpupy, a.pt, b.pt, out.pt)
    else: return array(None, lib.funcs.fTdot(p_gpupy, a.pt,b.pt))
    
def dotT(a,b,out=None):
    if out: lib.funcs.inp_dotT(p_gpupy, a.pt, b.pt, out.pt)
    else: return array(None, lib.funcs.fdotT(p_gpupy, a.pt,b.pt))
    
def fsynchronizingAdd(x1, out=None): 
    if out: lib.funcs.inp_synchronizingAdd(p_gpupy, x1.pt, out.pt)
    return array(None, lib.funcs.fsynchronizingAdd(p_gpupy,x1.pt))  

def dropout(x1,rate, out=None):
    if out: lib.funcs.inp_dropout(p_gpupy, x1.pt, out.pt, ct.c_float(rate))
    else: return array(None, lib.funcs.fdropout(p_gpupy, x1.pt, ct.c_float(rate)))
