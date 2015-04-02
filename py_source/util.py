'''
Created on Mar 24, 2015

@author: tim
'''
import numpy as np

def handle_shape(shape):    
    assert len(shape)<=4, "GPUpy only supports up tp four dimensions!"
    if len(shape) == 1: shape = (1,1,1) + shape
    if len(shape) == 2: shape = (1,1) + shape
    if len(shape) == 3: shape = (1,) + shape
    return shape 
    
def handle_dim(d0, d1, d2, d3):
    if d1 == None: d3 = d0; d2=1; d1=1; d0=1
    if d2 == None: d3 = d1; d2=d0; d1=1; d0=1
    if d3 == None: d3 = d2; d2=d1; d1=d0; d0=1
    return d0,d1,d2,d3

def handle_selectors(selectors):
    full_slice = slice(0,np.iinfo(np.int32).max)
    print type(selectors)
    if type(selectors) == type(slice(1)): selectors = [selectors]
    selects = []
    for i, slice_obj in enumerate(selectors):
        if not slice_obj.start and not slice_obj.stop: selects.append(slice(0,np.iinfo(np.int32).max))
        elif not slice_obj.start and slice_obj.stop: selects.append(slice(0,slice_obj.stop))
        elif slice_obj.start and not slice_obj.stop: selects.append(slice(slice_obj.start,np.iinfo(np.int32).max))
        else: selects.append(slice_obj)
        
    if len(selects) == 4: return selects
    if len(selects) == 3: return [full_slice] + selects
    if len(selects) == 2: return [full_slice,full_slice] + selects
    if len(selects) == 1: return [full_slice,full_slice, full_slice, selects[0]]