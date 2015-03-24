'''
Created on Mar 22, 2015

@author: tim
'''
import ctypes as ct
from gpupy import Tensor
from gpupy import array
import util as u
import numpy as np
lib = ct.cdll.LoadLibrary('./gpupylib.so')


lib.frand.restype = ct.POINTER(Tensor)
lib.frandn.restype = ct.POINTER(Tensor)
lib.fnormal.restype = ct.POINTER(Tensor)




class RandomState(object):
    def __init__(self, seed = None):
        if seed: self.p_gpupy = lib.fseeded_GPUpy(seed)
        else: self.p_gpupy = lib.fGPUpy()
        
        
    def rand(self,d0,d1=None,d2=None,d3=None):
        d0, d1, d2, d3 = u.handle_dim(d0, d1, d2, d3)
        return array(None, lib.frand(self.p_gpupy, d0,d1,d2,d3))
    
    def randn(self,d0,d1=None,d2=None,d3=None):
        d0, d1, d2, d3 = u.handle_dim(d0, d1, d2, d3)
        return array(None, lib.frandn(self.p_gpupy, d0, d1,d2,d3))
    
    def normal(self, loc=0.0, scale = 1.0, size=None):             
        d0, d1, d2, d3 = u.handle_shape(size)
        assert (d0*d1*d2*d3) % 2 == 0, "Size must be a multiple of 2!"
        assert size,"Size must be greater than zero!" 
        return array(None, lib.fnormal(self.p_gpupy, d0, d1,d2,d3,ct.c_float(loc),ct.c_float(scale)))
    

        
        
rdm = RandomState()          
def rand(d0,d1=None,d2=None,d3=None): return rdm.rand(d0,d1,d2,d3)
def randn(d0,d1=None,d2=None,d3=None): return rdm.randn(d0,d1,d2,d3)
def normal(loc=0.0, scale=1.0, size=None): return rdm.normal(loc,scale,size)        