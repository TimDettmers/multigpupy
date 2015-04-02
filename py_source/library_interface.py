'''
Created on Apr 2, 2015

@author: tim
'''
import ctypes as ct
funcs = ct.cdll.LoadLibrary('./gpupylib.so')

class Tensor(ct.Structure):
    _fields_ = [('batches', ct.c_int),
                ('maps', ct.c_int),
                ('rows', ct.c_int),
                ('cols', ct.c_int),
                ('bytes', ct.c_size_t),
                ('size', ct.c_int),
                ('data', ct.POINTER(ct.c_float))]    
    def __init__(self): pass

funcs.fempty.restype = ct.POINTER(Tensor)
funcs.fzeros.restype = ct.POINTER(Tensor)
funcs.fones.restype = ct.POINTER(Tensor)
funcs.ftocpu.restype = ct.POINTER(Tensor)
funcs.ftogpu.restype = ct.c_void_p
funcs.ffree.restype = ct.c_void_p
funcs.fT.restype = ct.POINTER(Tensor)
funcs.inp_T.restype = ct.c_void_p

funcs.fadd.restype = ct.POINTER(Tensor)
funcs.inp_add.restype = ct.c_void_p
funcs.fsub.restype = ct.POINTER(Tensor)
funcs.inp_sub.restype = ct.c_void_p
funcs.fmul.restype = ct.POINTER(Tensor)
funcs.inp_mul.restype = ct.c_void_p
funcs.fdiv.restype = ct.POINTER(Tensor)
funcs.inp_div.restype = ct.c_void_p
funcs.fscalarAdd.restype = ct.POINTER(Tensor)
funcs.inp_scalarAdd.restype = ct.c_void_p
funcs.fscalarSub.restype = ct.POINTER(Tensor)
funcs.inp_scalarSub.restype = ct.c_void_p
funcs.fscalarMul.restype = ct.POINTER(Tensor)
funcs.inp_scalarMul.restype = ct.c_void_p
funcs.fscalarDiv.restype = ct.POINTER(Tensor)
funcs.inp_scalarDiv.restype = ct.c_void_p

funcs.fexp.restype = ct.POINTER(Tensor)
funcs.inp_exp.restype = ct.c_void_p
funcs.flog.restype = ct.POINTER(Tensor)
funcs.inp_log.restype = ct.c_void_p
funcs.fsqrt.restype = ct.POINTER(Tensor)
funcs.inp_sqrt.restype = ct.c_void_p
funcs.flogistic.restype = ct.POINTER(Tensor)
funcs.inp_logistic.restype = ct.c_void_p
funcs.flogisticGrad.restype = ct.POINTER(Tensor)
funcs.inp_logisticGrad.restype = ct.c_void_p
funcs.ffabs.restype = ct.POINTER(Tensor)
funcs.inp_abs.restype = ct.c_void_p
funcs.fsquare.restype = ct.POINTER(Tensor)
funcs.inp_square.restype = ct.c_void_p
funcs.ffpow.restype = ct.POINTER(Tensor)
funcs.inp_pow.restype = ct.c_void_p

funcs.faddVectorToTensor.restype = ct.POINTER(Tensor)
funcs.inp_addVectorToTensor.restype = ct.c_void_p
funcs.fsubVectorToTensor.restype = ct.POINTER(Tensor)
funcs.inp_subVectorToTensor.restype = ct.c_void_p
funcs.fmulVectorToTensor.restype = ct.POINTER(Tensor)
funcs.inp_mulVectorToTensor.restype = ct.c_void_p
funcs.fdivVectorToTensor.restype = ct.POINTER(Tensor)
funcs.inp_divVectorToTensor.restype = ct.c_void_p

funcs.feq.restype = ct.POINTER(Tensor)
funcs.inp_eq.restype = ct.c_void_p
funcs.fls.restype = ct.POINTER(Tensor)
funcs.inp_ls.restype = ct.c_void_p
funcs.fgt.restype = ct.POINTER(Tensor)
funcs.inp_gt.restype = ct.c_void_p
funcs.fle.restype = ct.POINTER(Tensor)
funcs.inp_le.restype = ct.c_void_p
funcs.fge.restype = ct.POINTER(Tensor)
funcs.inp_ge.restype = ct.c_void_p
funcs.fne.restype = ct.POINTER(Tensor)
funcs.inp_ne.restype = ct.c_void_p



funcs.frand.restype = ct.POINTER(Tensor)
funcs.frandn.restype = ct.POINTER(Tensor)
funcs.fnormal.restype = ct.POINTER(Tensor)

funcs.fseeded_GPUpy.restype = ct.c_void_p
funcs.fGPUpy.restype = ct.c_void_p

class lib(object): funcs = funcs




    

