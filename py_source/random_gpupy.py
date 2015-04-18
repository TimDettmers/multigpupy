'''
Created on Mar 22, 2015

@author: tim
'''
import ctypes as ct
from library_interface import Tensor, lib
from gpupy import array
import util as u
import numpy as np


class RandomState(object):
    def __init__(self, seed = None):
        ptr = lib.funcs.fempty_split(1,1,1,128,-1)    
        if seed: self.p_gpupy = lib.funcs.fseeded_GPUpy(seed,lib.floats_8bit.ctypes.data_as(ct.POINTER(ct.c_float)))
        else: self.p_gpupy = lib.funcs.fGPUpy(lib.floats_8bit.ctypes.data_as(ct.POINTER(ct.c_float)))
        
        
    def rand(self,d0,d1=None,d2=None,d3=None):
        d0, d1, d2, d3 = u.handle_dim(d0, d1, d2, d3)
        return array(None, lib.funcs.frand(self.p_gpupy, d0,d1,d2,d3))
    
    def randn(self,d0,d1=None,d2=None,d3=None):
        d0, d1, d2, d3 = u.handle_dim(d0, d1, d2, d3)
        return array(None, lib.funcs.frandn(self.p_gpupy, d0, d1,d2,d3))
    
    def normal(self, loc=0.0, scale = 1.0, size=None):             
        d0, d1, d2, d3 = u.handle_shape(size)
        assert (d0*d1*d2*d3) % 2 == 0, "Size must be a multiple of 2!"
        assert size,"Size must be greater than zero!" 
        return array(None, lib.funcs.fnormal(self.p_gpupy, d0, d1,d2,d3,ct.c_float(loc),ct.c_float(scale)))    
        
rdm = RandomState()          
def rand(d0,d1=None,d2=None,d3=None): return rdm.rand(d0,d1,d2,d3)
def randn(d0,d1=None,d2=None,d3=None): return rdm.randn(d0,d1,d2,d3)
def normal(loc=0.0, scale=1.0, size=None): return rdm.normal(loc,scale,size)        