'''
Created on Mar 21, 2015

@author: tim
'''

import numpy as np
import ctypes as ct
import util as u
from library_interface import lib
import memory_handler as h
import uuid

mem = h.MemoryHandler()

ptr = lib.funcs.fempty_split(1,1,1,128,-1)
p_gpupy = lib.funcs.fGPUpy(lib.floats_8bit.ctypes.data_as(ct.POINTER(ct.c_float)))

def deprecated(func):
    fname = func.func_name
    print 'Function {0} is deprecated'.format(fname)
    return func

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
                     

class Array(object):
    def __init__(self, npArray = None, mat_pointer = None, split_idx=-1, dtype=np.float32):
        self.shape = None
        self.dummy = False
        self.id = uuid.uuid4()
        self.split_idx=split_idx         
                     
        if type(npArray) == type(np.array(1)):
            npArray = np.float32(npArray)
            shape = u.handle_shape(npArray.shape)
            
            mat_pointer = lib.funcs.fempty_split(shape[0],shape[1],shape[2],shape[3],split_idx)
            lib.funcs.ftogpu_split(mat_pointer, npArray.ctypes.data_as(ct.POINTER(ct.c_float)),split_idx)
            self.shape = npArray.shape 
            
        if mat_pointer:             
            self.shape_tensor = (mat_pointer.contents.batches, mat_pointer.contents.maps,
                                 mat_pointer.contents.rows, mat_pointer.contents.cols)
            if not self.shape: self.shape = self.shape_tensor
            
               
        self.pt = mat_pointer
        self.npArray = npArray   
        mem.arrays[self.id] = [self.shape_tensor, self.pt, dtype]   
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
        return array(None, lib.funcs.fslice(self.pt,S.pt))
        pass
    
    
    def tocpu(self):
        if self.split_idx == -1: 
            data = np.empty(self.shape, dtype=np.float32)
        elif self.split_idx == 2:
            data = np.empty((self.shape_tensor[0], self.shape_tensor[1], self.shape_tensor[2]/gpu_count(), self.shape_tensor[3]), dtype=np.float32)
        lib.funcs.ftocpu(self.pt, data.ctypes.data_as(ct.POINTER(ct.c_float)))   
        if data.shape[0] == 1 and data.shape[1] == 1 and data.shape[2] == 1: data = data.reshape(data.shape[3])
        if data.shape[0] == 1 and data.shape[1] == 1: data = data.reshape(data.shape[2], data.shape[3])
        if data.shape[0] == 1 and data.shape[1] > 1: data = data.reshape(data.shape[1], data.shape[2], data.shape[3])
        
        self.npArray = data
        
        return data        
    
    
    def sum(self): return lib.funcs.fsum(self.pt)
    def min(self): return lib.funcs.ffmin(self.pt)
    def max(self): return lib.funcs.ffmax(self.pt)

    @property
    def T(self): return array(None, lib.funcs.fT(self.pt))   
      
    def __del__(self): 
        lib.funcs.ffree(self.pt) 
        if mem:
            del mem.arrays[self.id]
    def __add__(self, other): return add(self, other)
    def __sub__(self, other): return apply_func(self,other, lib.funcs.fsub, lib.funcs.fscalarSub, lib.funcs.fsubVectorToTensor)
    def __mul__(self, other): return apply_func(self,other, lib.funcs.fmul, lib.funcs.fscalarMul, lib.funcs.fmulVectorToTensor)
    def __div__(self, other): return apply_func(self,other, lib.funcs.fdiv, lib.funcs.fscalarDiv, lib.funcs.fdivVectorToTensor)
    def __eq__(self, other): return equal(self,other)
    def __lt__(self, other):  return less(self,other)
    def __gt__(self, other): return greater(self,other)
    def __ge__(self, other): return greater_equal(self,other)
    def __le__(self, other): return less_equal(self,other)
    def __ne__(self, other): return not_equal(self,other)
    def __pow__(self, other): return power(self,other)
    #def abs(self): return absolute(self, out=None)
    def __str__(self): 
        printmat(self)
        return ''
    
    
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
    
def array(npArray = None, mat_pointer = None, split_idx=-1): 
    return Array(npArray, mat_pointer, split_idx)
    
def apply_func(x1, x2, func_matrix, func_scalar, func_vector = None, out=None): 
    is_scalar =  isinstance(x2, int) or isinstance(x2, float)
    if not is_scalar: 
        is_vectors = [is_vector(x1), is_vector(x2)]   
    if is_scalar: 
        if out: func_scalar(x1.pt,ct.c_float(x2),out.pt)
        else: return array(None,func_scalar(x1.pt,ct.c_float(x2)))
    elif any(is_vectors) and not all(is_vectors): 
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

def zeros(shape, split_idx=-1):
    shape = u.handle_shape(shape)
    out = array(None, lib.funcs.fzeros_split(shape[0],shape[1],shape[2],shape[3], split_idx))
    return out


def ones(shape):
    shape = u.handle_shape(shape)
    out = array(None, lib.funcs.fones(shape[0],shape[1],shape[2],shape[3]))
    return out

def empty(shape,split_idx=-1):
    shape = u.handle_shape(shape)
    out = array(None, lib.funcs.fempty_split(shape[0],shape[1],ct.c_int32(shape[2]),ct.c_int32(shape[3]),split_idx), split_idx)
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
    
def linear(x1,out=None):
    if out: copy(x1,out)
    else: return copy(x1,out)
    
def copy(x1,out=None):
    if out: lib.funcs.inp_copy(x1.pt,out.pt);
    else: return array(None, lib.funcs.fcopy(x1.pt))
    
def logistic(x1,out=None):
    if out: lib.funcs.inp_logistic(x1.pt,out.pt);
    else: return array(None, lib.funcs.flogistic(x1.pt))    
    
def double_ReLU(x1,out=None):
    if out: lib.funcs.inp_double_ReLU(x1.pt,out.pt);
    else: return array(None, lib.funcs.fdouble_ReLU(x1.pt))    
    
def double_ReLU_grad(x1,out=None):
    if out: lib.funcs.inp_double_ReLU_grad(x1.pt,out.pt);
    else: return array(None, lib.funcs.fdouble_ReLU_grad(x1.pt))

def logistic_grad(x1,out=None):    
    """Computes x1*(1-x1).
    :x1: Logistic Input.
    :out: Write output to this Tensor.
    """
    if out: lib.funcs.inp_logistic_grad(x1.pt,out.pt);
    else: return array(None, lib.funcs.flogistic_grad(x1.pt))
    
def rectified_linear(x1,out=None):
    if out: lib.funcs.inp_ReLU(x1.pt,out.pt);
    else: return array(None, lib.funcs.fReLU(x1.pt))
    
def ReLU(x1,out=None): 
    if out: rectified_linear(x1, out)
    else: return rectified_linear(x1, out)

def ReLU_grad(x1,out=None):    
    """Computes x1>0 elementwise
    :x1: Logistic Input.
    :out: Write output to this Tensor.
    """
    if out: greater(x1,0.0,out)
    else: return greater(x1,0.0,out)

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
    if x2 == None: return False
    elif out: apply_func(x1,x2, lib.funcs.inp_eq, lib.funcs.inp_scalar_eq, lib.funcs.inp_vec_eq, out)
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
    if x2 == None: return False
    elif out: apply_func(x1,x2, lib.funcs.inp_ne, lib.funcs.inp_scalar_ne, lib.funcs.inp_vec_ne, out)
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

def transpose(x1, out=None):
    if out: lib.funcs.inp_T(x1.pt,out.pt);
    else: return array(None, lib.funcs.fT(x1.pt))

def print_tensor(x1):
    lib.funcs.ffprint(x1.pt)

def enable_peer_access():
    lib.funcs.fenablePeerAccess(p_gpupy)    
    
def disable_peer_access():
    lib.funcs.fdisablePeerAccess(p_gpupy)  

def dropout(x1,rate, out=None):
    if out: lib.funcs.inp_dropout(p_gpupy, x1.pt, out.pt, ct.c_float(rate))
    else: return array(None, lib.funcs.fdropout(p_gpupy, x1.pt, ct.c_float(rate)))
    
def softmax(x1,out=None):
    if out: lib.funcs.inp_softmax(x1.pt, out.pt)
    return array(None, lib.funcs.fsoftmax(x1.pt))  

def argmax(x1,out=None):
    if out: lib.funcs.inp_argmax(x1.pt, out.pt)
    return array(None, lib.funcs.fargmax(x1.pt))  

def slice_axis(A, out):
    lib.funcs.inp_slice_axis(A.pt, out.pt)
    
def stack_axis(A, out):
    lib.funcs.inp_stack_axis(A.pt, out.pt)
    
def sync(source, layer_idx = 0, dtype=np.float32):    
    arrays = mem.get_arrays_for_sync(source, dtype)
    if len(arrays) == 4: lib.sync_func[dtype](p_gpupy, source.pt, arrays[0].pt, arrays[1].pt, arrays[2].pt, layer_idx)
    elif len(arrays) == 3:  lib.sync_func[dtype](p_gpupy, source.pt, arrays[0].pt, arrays[1].pt, None, layer_idx)
    else: lib.sync_func[dtype](p_gpupy, source.pt, arrays[0].pt, None, None, layer_idx)  
    
def compress_and_sync(source, layer_idx = 0, abs_max_value=1.0, dtype=np.char):
    if dtype==np.char:
        if source.id not in mem.compression_arrays: mem.compression_arrays[source.id] = [empty_char_like(source), empty_like(source)]
        compress_8bit(source, abs_max_value, mem.compression_arrays[source.id][0])
        sync(mem.compression_arrays[source.id][0], layer_idx, dtype)
    if dtype==np.float32: sync(source, layer_idx, dtype)
        
def decompress_sync_streams_add(source, layer_idx = 0, split_idx=2, abs_max_value=1.0, dtype=np.char):
    if dtype==np.char: 
        lib.funcs.fsynchronize_streams(p_gpupy,layer_idx)
        arrays = mem.sync_arrays[mem.compression_arrays[source.id][0].id]    
        decompress_8bit(arrays[0], abs_max_value,mem.compression_arrays[source.id][1])
        add(source, mem.compression_arrays[source.id][1],source)
        if len(arrays) > 1: 
            decompress_8bit(arrays[1], abs_max_value,mem.compression_arrays[source.id][1])
            add(source, mem.compression_arrays[source.id][1],source)
        if len(arrays) > 2: 
            decompress_8bit(arrays[2], abs_max_value,mem.compression_arrays[source.id][1])
            add(source, mem.compression_arrays[source.id][1],source)
        return source
    
    elif dtype == np.float32: sync_streams_add(source, layer_idx, split_idx)
   
        
        
    
            
def sync_streams(layer_idx=0): lib.funcs.fsynchronize_streams(p_gpupy,layer_idx)
def sync_streams_add(source, layer_idx=0, split_idx=2):
    lib.funcs.fsynchronize_streams(p_gpupy,layer_idx)    
    arrays = mem.sync_arrays[source.id]    
    add(source, arrays[0], source)
    if len(arrays) > 1: add(arrays[1], source, source)
    if len(arrays) > 2: add(arrays[2], source, source)
    return source
        

def sum(x1): return lib.funcs.fsum(x1.pt)
def min(x1): return lib.funcs.ffmin(x1.pt)
def max(x1): return lib.funcs.ffmax(x1.pt)

def print_free_memory():
    return lib.funcs.fprint_free_memory()

def is_synchronizing(): return lib.funcs.fis_synchronizing()
def current_sync_idx(): return lib.funcs.fcurrent_sync_idx()
def reset_sync_idx(): lib.funcs.freset_sync_idx()

def create_additional_streams(layer_count):    
    lib.funcs.fcreate_streams(p_gpupy, layer_count)
    
def zeros_like(x1):
    arr = array(None, lib.funcs.fempty_like(x1.pt))
    lib.funcs.ffill(arr.pt, ct.c_float(0.0))
    return arr
    
def empty_like(x1): return array(None, lib.funcs.fempty_like(x1.pt))
def empty_uint_like(x1): return array(None, lib.funcs.fempty_uint_like(x1.pt))
def empty_char_like(x1): return array(None, lib.funcs.fempty_char_like(x1.pt))
def empty_ushort_like(x1): return array(None, lib.funcs.fempty_ushort_like(x1.pt))

def compress_8bit(A, abs_max_value, char_tensor):lib.funcs.fcompress_8bit(p_gpupy, A.pt,ct.c_float(abs_max_value), char_tensor.pt)
def decompress_8bit(char_tensor, abs_max_value, out):lib.funcs.fdecompress_8bit(p_gpupy, char_tensor.pt,ct.c_float(abs_max_value),out.pt)   
def compress_16bit(A, out): lib.funcs.fcompress_16bit(A.pt, out.pt)
def decompress_16bit(A, out): lib.funcs.fdecompress_16bit(A.pt, out.pt) 

def sum_row(x1, out): lib.funcs.frow_sum(x1.pt, out.pt)
def max_row(x1, out): lib.funcs.frow_max(x1.pt, out.pt)
    
def compress_1bit(A, val_with_errors, errors, avgPositive,  avgNegative, out, maskPos, maskNeg, posCount, negCount):
    add(A,errors,val_with_errors)
    #print val_with_errors.sum(), A.sum(), errors.sum()
    greater_equal(val_with_errors, 0.0, maskPos)
    less(val_with_errors, 0.0, maskNeg)
    sum_row(maskPos, posCount)
    fill(negCount, A.shape_tensor[3])
    sub(negCount,posCount, negCount)
    mul(val_with_errors,maskPos, maskPos)
    mul(val_with_errors,maskNeg, maskNeg)
    sum_row(maskPos, avgPositive)
    sum_row(maskNeg, avgNegative)    
    div(avgPositive, posCount, avgPositive)
    div(avgNegative, negCount, avgNegative)
    #print A.sum(), errors.sum(), avgPositive.sum(), avgNegative.sum()
    lib.funcs.fcompress_1bit(A.pt, errors.pt, avgPositive.pt, avgNegative.pt, out.pt)
    
def decompress_1bit(quant, errors, avgPos, avgNeg, out):
    lib.funcs.fdecompress_1bit(quant.pt, errors.pt, avgPos.pt, avgNeg.pt, out.pt)
    
def fill(x1, fill_value): lib.funcs.ffill(x1.pt, ct.c_float(fill_value))
     
def tick(eventname='default'): lib.funcs.ftick(p_gpupy, ct.c_char_p(eventname))
def tock(eventname='default'): return lib.funcs.ftock(p_gpupy, ct.c_char_p(eventname))


def to_pinned_pointer(x1):
    if x1.dtype != np.float32: x1 = np.float32(x1)
    shape = u.handle_shape(x1.shape)
    return lib.funcs.fto_pinned(shape[0],shape[1], shape[2], shape[3], x1.ctypes.data_as(ct.POINTER(ct.c_float)))    

def empty_pinned_pointer(shape):      
    out = np.empty(shape)
    shape_tensor = u.handle_shape(shape)
    return lib.funcs.fto_pinned(shape_tensor[0],shape_tensor[1], shape_tensor[2], shape_tensor[3], out.ctypes.data_as(ct.POINTER(ct.c_float)))  

def pointer2ndarray(pt, shape, dtype=np.float32):
    shape_tensor = u.handle_shape(shape)
    size = shape_tensor[0]*shape_tensor[1]*shape_tensor[2]*shape_tensor[3]
    str_buffer = ct.string_at(pt, ct.sizeof(ct.c_float)*size)
    return np.fromstring(str_buffer, dtype=dtype).reshape(shape)   

def to_col_major_pinned_pointer(x1, pt_out=None):
    if x1.dtype != np.float32: x1 = np.float32(x1)
    shape = u.handle_shape(x1.shape)
    if not pt_out: pt_out = empty_pinned_pointer(shape) 
    lib.funcs.inp_to_col_major_pinned(x1.ctypes.data_as(ct.POINTER(ct.c_float)),pt_out, shape[0],shape[1],shape[2],shape[3])
    return pt_out

def printmat(x1): printfull(x1, 0, x1.shape_tensor[2], 0, x1.shape_tensor[3])
def printrows(x1, start_rows, end_rows): printfull(x1, start_rows, end_rows, 0, x1.shape_tensor[3])
def printfull(x1, start_rows, end_rows, start_cols, end_cols):
    lib.funcs.fprintmat(x1.pt, start_rows, end_rows, start_cols, end_cols)
    pass
    


    

    
    
