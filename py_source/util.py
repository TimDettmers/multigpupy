'''
Created on Mar 24, 2015

@author: tim
'''



def handle_shape(shape):
    assert(len(shape)>4, "GPUpy only supports up tp four dimensions!")
    if len(shape) == 1: shape = (1,1,1) + shape
    if len(shape) == 2: shape = (1,1) + shape
    if len(shape) == 3: shape = (1,) + shape
    return shape 
    
def handle_dim(d0, d1, d2, d3):
    if d1 == None: d3 = d0; d2=1; d1=1; d0=1
    if d2 == None: d3 = d1; d2=d0; d1=1; d0=1
    if d3 == None: d3 = d2; d2=d1; d1=d0; d0=1
    return d0,d1,d2,d3