
'''
Created on Mar 21, 2015

@author: tim
'''
import nose
import gpupy as gpu
import random_gpupy as rdm
import numpy as np
import numpy.testing as t
from batch_allocator import batch_allocator
import time
import util as u
from layer import *

def setup():
    pass

def teardown():
    pass

def test_cpu():
    A = np.random.rand(17,83)
    B = gpu.array(A)
    C = B.tocpu()
    t.assert_array_almost_equal(A,C,4,"array.tocpu not equal to init array!")
    pass

def test_rand_seed():
    A = rdm.RandomState(12345)
    B = rdm.RandomState(12345)
    
    C1 = A.rand(5090,83)
    C2 = B.rand(5090,83)
    t.assert_array_almost_equal(C1.tocpu(),C2.tocpu(),6,"Rand seed does not give unique results!")
    t.assert_(C1.tocpu().sum() > 1.0,"Array has too many zero values!")
    

def test_uniform():
    rdmstate = rdm.RandomState(17)
    A = rdmstate.rand(83,17,7)
    B = rdm.rand(17,83,7)
    
    C1 = A.tocpu()
    C2 = B.tocpu()    

    t.assert_allclose(np.mean(C1), 0.5, rtol=0.05, atol=0.05, err_msg="Poor mean value for uniform distribution!")
    t.assert_allclose(np.mean(C2), 0.5, rtol=0.05, atol=0.05, err_msg="Poor mean value for uniform distribution!")
    t.assert_allclose(np.var(C1), 0.08333, rtol=0.05, atol=0.01, err_msg="Poor variance value for uniform distribution!")
    t.assert_allclose(np.var(C2), 0.08333, rtol=0.05, atol=0.01, err_msg="Poor variance value for uniform distribution!")
    
    assert np.max(C1) <= 1.0,"Max value larger than 1.0"
    assert np.max(C2) <= 1.0,"Max value larger than 1.0"
    assert np.min(C1) > 0.0,"Min value smaller than 0.0"
    assert np.min(C2) > 0.0,"Min value smaller than 0.0"
    
def test_normal():
    rdmstate = rdm.RandomState(17)
    A = rdmstate.normal(1,5,(830,17))
    B = rdm.normal(-17,1,(17,830))
    
    C1 = A.tocpu()
    C2 = B.tocpu()    

    t.assert_allclose(np.mean(C1), 1.0, rtol=0.05, atol=0.1, err_msg="Poor mean value for normal distribution!")
    t.assert_allclose(np.mean(C2), -17, rtol=0.05, atol=0.05, err_msg="Poor mean value for normal distribution!")
    t.assert_allclose(np.std(C1), 5.0, rtol=0.05, atol=0.25, err_msg="Poor standard deviation value for normal distribution!")
    t.assert_allclose(np.std(C2), 1.0, rtol=0.05, atol=0.05, err_msg="Poor standard deviation value for normal distribution!")
    
    #one std should contain about 68% of all values   
    C3 = np.sum((C1 >6.0)+(C1 <-4.0))/float(C1.size)
    C4 = np.sum((C2 >-16.0)+(C2 <-18.0))/float(C1.size)
    t.assert_allclose(C3, 0.32, rtol=0.2, atol=0.05, err_msg="68% of values should be within one standard deviation.")
    t.assert_allclose(C4, 0.32, rtol=0.2, atol=0.05, err_msg="68% of values should be within one standard deviation.")
    
def test_transpose():    
    A = np.float32(np.random.rand(10,10))
    C = gpu.array(A).T.tocpu()
    t.assert_array_equal(A.T, C, "Transpose not identical to numpy")    
    
    A = np.float32(np.random.rand(17,17,17))
    C1 = gpu.array(A).T.tocpu()
    C2 = np.transpose(A,(0,2,1))
    t.assert_array_equal(C1, C2, "Transpose not identical to numpy")
    
    A = np.float32(np.random.rand(17,17,17,17))
    C1 = gpu.array(A).T.tocpu()
    C2 = np.transpose(A,(0,1,3,2))
    t.assert_array_equal(C1, C2, "Transpose not identical to numpy")
    

   
   
def test_togpu():
    A = np.float32(np.random.rand(1333,177))
    C = gpu.array(A).tocpu()    
    t.assert_array_equal(A, C, "To GPU does not work!")
    
    A = np.float32(np.random.rand(177,17,17))    
    C = gpu.array(A).tocpu()
    t.assert_array_equal(A, C, "To GPU does not work!")

    A = np.float32(np.random.rand(2,17,17,17))    
    C = gpu.array(A).tocpu()
    t.assert_array_equal(A, C, "To GPU does not work!")
    
    A = np.float32(np.random.rand(2,17,1,17))    
    C = gpu.array(A).tocpu()
    t.assert_array_equal(A, C, "To GPU does not work!")
    
    
    
def test_add():
    A = np.random.rand(10,7,83,4)
    B = np.random.rand(10,7,83,4)
    v = np.float32(np.random.rand(4))
    w = gpu.array(v)
    C1 = gpu.array(A)
    C2 = gpu.array(B)
    C = gpu.add(C1,C2)  

    out = gpu.empty(C.shape)      
    gpu.add(C1, C2, out)
    t.assert_array_almost_equal(C.tocpu(), A+B, 7, "Add not equal to numpy add!")
    t.assert_array_almost_equal(out.tocpu(), A+B, 7, "Add not equal to numpy add!")
    t.assert_array_almost_equal((C1+C2).tocpu(), A+B, 7, "Add not equal to numpy add!")    
    
    C = gpu.add(C1,w)  
    out = gpu.empty(C.shape)      
    gpu.add(C1, w, out)
    t.assert_array_almost_equal(C.tocpu(), A+v, 7, "Vector add not equal to numpy add!")
    t.assert_array_almost_equal(out.tocpu(), A+v, 7, "Vector add not equal to numpy add!")        
    t.assert_array_almost_equal((C1+w).tocpu(), A+v, 7, "Vector add not equal to numpy add!")
    t.assert_array_almost_equal((w+C1).tocpu(), v+A, 7, "Vector add not equal to numpy add!")
    
    C1+=C2
    t.assert_array_almost_equal(C1.tocpu(), A+B, 7, "Add not equal to numpy add!")
    C1 = gpu.array(A)
    C1+=w
    t.assert_array_almost_equal(C1.tocpu(), A+v, 7, "Add not equal to numpy add!")    
    
 
def test_sub():
    A = np.random.rand(10,7,83,4)
    B = np.random.rand(10,7,83,4)
    v = np.float32(np.random.rand(4))
    w = gpu.array(v)
    C1 = gpu.array(A)
    C2 = gpu.array(B)
    C = gpu.sub(C1,C2)  
    out = gpu.empty(C.shape)      
    gpu.sub(C1, C2, out)
    t.assert_array_almost_equal(C.tocpu(), A-B, 7, "Add not equal to numpy sub!")
    t.assert_array_almost_equal(out.tocpu(), A-B, 7, "Add not equal to numpy sub!")
    t.assert_array_almost_equal((C1-C2).tocpu(), A-B, 7, "Add not equal to numpy sub!")
    
    C = gpu.sub(C1,w)  
    out = gpu.empty(C.shape)      
    gpu.sub(C1, w, out)
    t.assert_array_almost_equal(C.tocpu(), A-v, 7, "Vector sub not equal to numpy sub!")
    t.assert_array_almost_equal(out.tocpu(), A-v, 7, "Vector sub not equal to numpy sub!")        
    t.assert_array_almost_equal((C1-w).tocpu(), A-v, 7, "Vector sub not equal to numpy sub!")
    #TODO: Add scaling constant to vector-matrix operation
    #t.assert_array_almost_equal((w-C1).tocpu(), v-A, 7, "Vector sub equal to numpy sub!")
    
    C1-=C2
    t.assert_array_almost_equal(C1.tocpu(), A-B, 7, "Sub not equal to numpy sub!")
    C1 = gpu.array(A)
    C1-=w
    t.assert_array_almost_equal(C1.tocpu(), A-v, 7, "Sub not equal to numpy sub!")   
    
def test_mul():
    A = np.random.rand(10,7,83,4)
    B = np.random.rand(10,7,83,4)
    v = np.float32(np.random.rand(4))
    w = gpu.array(v)
    C1 = gpu.array(A)
    C2 = gpu.array(B)
    C = gpu.mul(C1,C2)  
    out = gpu.empty(C.shape)      
    gpu.mul(C1, C2, out)
    t.assert_array_almost_equal(C.tocpu(), A*B, 7, "Add not equal to numpy mul!")
    t.assert_array_almost_equal(out.tocpu(), A*B, 7, "Add not equal to numpy mul!")
    t.assert_array_almost_equal((C1*C2).tocpu(), A*B, 7, "Add not equal to numpy mul!")
    
    C = gpu.mul(C1,w)  
    out = gpu.empty(C.shape)      
    gpu.mul(C1, w, out)
    t.assert_array_almost_equal(C.tocpu(), A*v, 7, "Vector mul not equal to numpy mul!")
    t.assert_array_almost_equal(out.tocpu(), A*v, 7, "Vector mul not equal to numpy mul!")        
    t.assert_array_almost_equal((C1*w).tocpu(), A*v, 7, "Vector mul not equal to numpy mul!")    
    t.assert_array_almost_equal((w*C1).tocpu(), v*A, 7, "Vector mul equal to numpy mul!")
    
    C1*=C2
    t.assert_array_almost_equal(C1.tocpu(), A*B, 7, "mul not equal to numpy mul!")
    C1 = gpu.array(A)
    C1*=w
    t.assert_array_almost_equal(C1.tocpu(), A*v, 7, "mul not equal to numpy mul!")  
    
def test_div():
    A = np.float32(np.random.rand(10,7,83,4))
    B = np.float32(np.random.rand(10,7,83,4))
    v = np.float32(np.random.rand(4))
    w = gpu.array(v)
    C1 = gpu.array(A)
    C2 = gpu.array(B)
    C = gpu.div(C1,C2)  
    out = gpu.empty(C.shape)      
    gpu.div(C1, C2, out)
    t.assert_array_almost_equal(C.tocpu(), A/B, 5, "Add not equal to numpy div!")
    t.assert_array_almost_equal(out.tocpu(), A/B, 5, "Add not equal to numpy div!")
    t.assert_array_almost_equal((C1/C2).tocpu(), A/B, 5, "Add not equal to numpy div!")
    
    C = gpu.div(C1,w)  
    out = gpu.empty(C.shape)      
    gpu.div(C1, w, out)
    t.assert_array_almost_equal(C.tocpu(), A/v, 5, "Vector div not equal to numpy div!")
    t.assert_array_almost_equal(out.tocpu(), A/v, 5, "Vector div not equal to numpy div!")        
    t.assert_array_almost_equal((C1/w).tocpu(), A/v, 5, "Vector div not equal to numpy div!")
    #TODO: 
    #t.assert_array_almost_equal((w/C1).tocpu(), v/A, 7, "Vector div equal to numpy div!")
    
    
    C = gpu.div(C1,C2)       
    gpu.div(C1, C2, out)
    t.assert_almost_equal(C.tocpu().sum(), (A/B).sum(), 5, "Add not equal to numpy div!")
    t.assert_almost_equal(out.tocpu().sum(), (A/B).sum(), 5, "Add not equal to numpy div!")
    t.assert_almost_equal((C1/C2).tocpu().sum(), (A/B).sum(), 5, "Add not equal to numpy div!")
    C1/=C2
    t.assert_array_almost_equal(C1.tocpu().sum(), (A/B).sum(), 5, "Add not equal to numpy div!")
 
def test_scalarAdd():
    A = np.float32(np.random.rand(10,7,83,4))
    flt = 17.83289
    B = gpu.array(A)
    C = gpu.add(B, flt).tocpu()    
    t.assert_array_equal(A+flt, C, "Scalar add not like numpy scalar add")
    t.assert_array_equal(A+flt, (B+flt).tocpu(), "Scalar add not like numpy scalar add")
    t.assert_array_equal(A+5, (B+5).tocpu(), "Scalar add not like numpy scalar add")
    B+=flt
    C = B.tocpu()
    t.assert_array_equal(A+flt, C, "Scalar add not like numpy scalar add")
   
def test_scalarSub():
    A = np.float32(np.random.rand(10,7,83,4))
    flt = 17.83289
    B = gpu.array(A)
    C = gpu.sub(B, flt).tocpu()    
    t.assert_array_equal(A-flt, C, "Scalar sub not like numpy scalar sub") 
    t.assert_array_equal(A-flt, (B-flt).tocpu(), "Scalar sub not like numpy scalar sub")
    t.assert_array_equal(A-5, (B-5).tocpu(), "Scalar sub not like numpy scalar sub")
    B-=flt
    C = B.tocpu()
    t.assert_array_equal(A-flt, C, "Scalar sub not like numpy scalar sub")  
    
def test_scalarMul():
    A = np.float32(np.random.rand(10,7,83,4))
    flt = 17.83289
    B = gpu.array(A)
    C = gpu.mul(B, flt).tocpu()
    t.assert_array_equal(A*flt, C, "Scalar mul not like numpy scalar mul") 
    t.assert_array_equal(A*flt, (B*flt).tocpu(), "Scalar mul not like numpy scalar mul") 
    t.assert_array_equal(A*5, (B*5).tocpu(), "Scalar mul not like numpy scalar mul") 
    B*=flt
    C = B.tocpu()
    t.assert_array_equal(A*flt, C, "Scalar mul not like numpy scalar mul") 
    
def test_scalarDiv():
    A = np.float32(np.random.rand(10,7,83,4))
    flt = 17.83289
    B = gpu.array(A)
    C = gpu.div(B, flt).tocpu()    
    t.assert_array_almost_equal(A/flt, C, 5, "Scalar div not like numpy scalar div") 
    t.assert_array_almost_equal(A/flt, (B/flt).tocpu(), 5, "Scalar div not like numpy scalar div")
    t.assert_array_almost_equal(A/5, (B/5).tocpu(), 5, "Scalar div not like numpy scalar div")
    B/=flt
    C = B.tocpu()
    t.assert_array_almost_equal(A/flt, C, 5, "Scalar div not like numpy scalar div") 
  
def test_exp():
    A = np.float32(np.random.rand(10,7,83,4))
    B = gpu.array(A)
    C = gpu.exp(B).tocpu()  
    t.assert_array_almost_equal(np.exp(A), C, 5, "Exp not like numpy scalar exp") 
    
def test_log():
    A = np.float32(np.random.rand(10,7,83,4))
    B = gpu.array(A)
    C = gpu.log(B).tocpu()  
    t.assert_array_almost_equal(np.log(A), C, 5, "Log not like numpy scalar log") 
    
def test_sqrt():
    A = np.float32(np.random.rand(10,7,83,4))
    B = gpu.array(A)
    C = gpu.sqrt(B).tocpu()  
    t.assert_array_almost_equal(np.sqrt(A), C, 5, "Log not like numpy scalar sqrt") 
    
def test_logistic():
    A = np.float32(np.random.rand(10,7,83,4))
    B = gpu.array(A)
    C = gpu.logistic(B).tocpu()  
    t.assert_array_almost_equal(1.0/(1.0+np.exp(-A)), C, 5, "Logistic not like numpy equivalent") 
    
def test_logisticGrad():
    A = np.float32(np.random.rand(10,7,83,4))
    B = gpu.array(A)
    C = gpu.logistic_grad(B).tocpu()  
    t.assert_array_almost_equal(C, A*(1.0-A), 5, "LogisticGrad not like numpy equivalent") 
    
def test_abs():    
    A = np.float32(np.random.rand(10,7,83,4))
    B = gpu.array(A)
    C = gpu.abs(B).tocpu()
    t.assert_array_almost_equal(C, np.absolute(A), 5, "abs not like numpy equivalent")
    
def test_square():
    A = np.float32(np.random.rand(10,7,83,4))
    B = gpu.array(A)
    C = gpu.square(B).tocpu()  
    t.assert_array_almost_equal(C, np.square(A), 5, "square not like numpy equivalent") 
    
def test_pow():
    A = np.float32(np.random.rand(10,7,83,4))
    B = gpu.array(A)
    C = gpu.power(B,5).tocpu()  
    t.assert_array_almost_equal(C, np.power(A,5), 5, "power not like numpy equivalent") 
    C = gpu.power(B,17.83).tocpu()  
    t.assert_array_almost_equal(C, np.power(A,17.83), 5, "power not like numpy equivalent") 
    
def test_addVectorToTensor():
    A1 = np.float32(np.random.rand(10,7,83,4))
    v1 = np.float32(np.random.rand(4))
    B1 = gpu.array(A1)
    b1 = gpu.array(v1)    
    
    C = gpu.add(B1, b1).tocpu()   
    t.assert_array_equal(C, A1+v1, "Vector Matrix addition not equal to numpy value")  
          
    gpu.add(B1, b1,B1)    
    t.assert_array_equal(B1.tocpu(), A1+v1, "Vector Matrix addition not equal to numpy value")  
 
def test_subVectorToTensor():
    A1 = np.float32(np.random.rand(10,7,83,4))
    v1 = np.float32(np.random.rand(4))
    B1 = gpu.array(A1)
    b1 = gpu.array(v1)    
    
    C = gpu.sub(B1, b1).tocpu()   
    t.assert_array_equal(C, A1-v1, "Vector Matrix addition not equal to numpy value")  
          
    gpu.sub(B1, b1,B1)    
    t.assert_array_equal(B1.tocpu(), A1-v1, "Vector Matrix addition not equal to numpy value")  
    
def test_mulVectorToTensor():
    A1 = np.float32(np.random.rand(10,7,83,4))
    v1 = np.float32(np.random.rand(4))
    B1 = gpu.array(A1)
    b1 = gpu.array(v1)    
    
    C = gpu.mul(B1, b1).tocpu()   
    t.assert_array_equal(C, A1*v1, "Vector Matrix addition not equal to numpy value")  
          
    gpu.mul(B1, b1,B1)    
    t.assert_array_equal(B1.tocpu(), A1*v1, "Vector Matrix addition not equal to numpy value")  
    
def test_divVectorToTensor():
    A1 = np.float32(np.random.rand(10,7,83,4))
    v1 = np.float32(np.random.rand(4))
    B1 = gpu.array(A1)
    b1 = gpu.array(v1)    
    
    C = gpu.div(B1, b1).tocpu()   
    t.assert_array_equal(C, A1/v1, "Vector Matrix addition not equal to numpy value")  
          
    gpu.div(B1, b1,B1)    
    t.assert_array_equal(B1.tocpu(), A1/v1, "Vector Matrix addition not equal to numpy value")  

    
def test_is_vectors():
    assert gpu.is_vector(gpu.empty((1,))) == True
    assert gpu.is_vector(gpu.empty((1,2))) == True
    assert gpu.is_vector(gpu.empty((2,2))) == False
    assert gpu.is_vector(gpu.empty((1,1,2))) == True
    assert gpu.is_vector(gpu.empty((1,2,2))) == False
    assert gpu.is_vector(gpu.empty((2,1,2))) == False
    assert gpu.is_vector(gpu.empty((1,1,1,2))) == True
    assert gpu.is_vector(gpu.empty((1,1,2,2))) == False
    assert gpu.is_vector(gpu.empty((1,2,1,2))) == False
    assert gpu.is_vector(gpu.empty((2,1,1,2))) == False
    assert gpu.is_vector(gpu.empty((2,2,1,2))) == False
    assert gpu.is_vector(gpu.empty((2,2,2,2))) == False
    
def test_eq():
    A1 = np.float32(np.random.rand(10,7,10,17))
    A2 = np.float32(np.random.rand(10,7,10,17))
    v = np.float32(np.random.rand(17))
    B1 = gpu.array(A1)
    B2 = gpu.array(A2)
    w = gpu.array(v)
    
    C1 = gpu.equal(B1,B1).tocpu()
    C2 = gpu.equal(B1,B2).tocpu()
    C3 = gpu.equal(B1,w).tocpu()
    C4 = gpu.equal(B1,0.51783).tocpu()
    
    t.assert_array_equal(C1, np.equal(A1,A1), "gpu.equal != np.equal")
    t.assert_array_equal(C2, np.equal(A1,A2), "gpu.equal != np.equal")
    t.assert_array_equal((B1==B1).tocpu(), A1==A1, "gpu == != np ==")
    t.assert_array_equal((B1==B2).tocpu(), A1==A2, "gpu == != np ==")
    
    t.assert_array_equal(C3, np.equal(A1,v), "vector gpu.equal != np.equal")
    t.assert_array_equal((B1==w).tocpu(), A1==v, "vector gpu == != np ==")
    t.assert_array_equal((w==B1).tocpu(), v==A1, "vector gpu == != np ==")
        
    t.assert_array_equal(C4, np.equal(A1,0.51783), "scalar gpu.equal != np.equal")
    t.assert_array_equal((B1==0.51783).tocpu(), A1==0.51783, "scalar gpu == != np ==")
    
 
def test_less():
    A1 = np.float32(np.random.rand(10,7,10,17))
    A2 = np.float32(np.random.rand(10,7,10,17))
    v = np.float32(np.random.rand(17))
    B1 = gpu.array(A1)
    B2 = gpu.array(A2)
    w = gpu.array(v)
    
    C1 = gpu.less(B1,B1).tocpu()
    C2 = gpu.less(B1,B2).tocpu()
    C3 = gpu.less(B1,w).tocpu()
    C4 = gpu.less(B1,0.51783).tocpu()
    
    t.assert_array_equal(C1, np.less(A1,A1), "gpu.less != np.less")
    t.assert_array_equal(C2, np.less(A1,A2), "gpu.less != np.less")
    t.assert_array_equal((B1<B1).tocpu(), A1<A1, "gpu < != np <")    
    t.assert_array_equal((B1<B2).tocpu(), A1<A2, "gpu < != np <")
        
    t.assert_array_equal(C3, np.less(A1,v), "vector gpu.less != np.less")    
    t.assert_array_equal((B1<w).tocpu(), A1<v, "vector gpu < != np <")
    #TODO: 
    #t.assert_array_equal((w<B1).tocpu(), v<A1, "vector gpu == != np ==")
    t.assert_array_equal(C4, np.less(A1,0.51783), "scalar gpu.less != np.less")
    t.assert_array_equal((B1<0.51783).tocpu(), A1<0.51783, "scalar gpu < != np <")
    
def test_greater():
    A1 = np.float32(np.random.rand(10,7,10,17))
    A2 = np.float32(np.random.rand(10,7,10,17))
    v = np.float32(np.random.rand(17))
    B1 = gpu.array(A1)
    B2 = gpu.array(A2)
    w = gpu.array(v)
    
    C1 = gpu.greater(B1,B1).tocpu()
    C2 = gpu.greater(B1,B2).tocpu()
    C3 = gpu.greater(B1,w).tocpu()
    C4 = gpu.greater(B1,0.51783).tocpu()
    
    t.assert_array_equal(C1, np.greater(A1,A1), "gpu.greater > np.greater")
    t.assert_array_equal(C2, np.greater(A1,A2), "gpu.greater > np.greater")
    t.assert_array_equal((B1>B1).tocpu(), A1>A1, "gpu > > np >")
    t.assert_array_equal((B1>B2).tocpu(), A1>A2, "gpu > > np >")   
    
    t.assert_array_equal(C3, np.greater(A1,v), "vector gpu.greater != np.greater")
    t.assert_array_equal((B1>w).tocpu(), A1>v, "vector gpu > != np >")
    
    t.assert_array_equal(C4, np.greater(A1,0.51783), "scalar gpu.greater != np.greater")
    t.assert_array_equal((B1>0.51783).tocpu(), A1>0.51783, "scalar gpu > != np >")
    
def test_less_equal():
    A1 = np.float32(np.random.rand(10,7,10,17))
    A2 = np.float32(np.random.rand(10,7,10,17))
    v = np.float32(np.random.rand(17))
    B1 = gpu.array(A1)
    B2 = gpu.array(A2)
    w = gpu.array(v)
    
    C1 = gpu.less_equal(B1,B1).tocpu()
    C2 = gpu.less_equal(B1,B2).tocpu()
    C3 = gpu.less_equal(B1,w).tocpu()
    C4 = gpu.less_equal(B1,0.51783).tocpu()
    
    t.assert_array_equal(C1, np.less_equal(A1,A1), "gpu.less_equal != np.less_equal")
    t.assert_array_equal(C2, np.less_equal(A1,A2), "gpu.less_equal != np.less_equal")
    t.assert_array_equal((B1<=B1).tocpu(), A1<=A1, "gpu <= != np <=")
    t.assert_array_equal((B1<=B2).tocpu(), A1<=A2, "gpu <= != np <=")   
    
    t.assert_array_equal(C3, np.less_equal(A1,v), "vector gpu.less_equal != np.less_equal")
    t.assert_array_equal((B1<=w).tocpu(), A1<=v, "vector gpu <= != np <=") 
    #TODO: 
    #t.assert_array_equal((w<=B1).tocpu(), v<=A1, "vector gpu <= != np <=") 
    
    t.assert_array_equal(C4, np.less_equal(A1,0.51783), "scalar gpu.less_equal != np.less_equal")
    t.assert_array_equal((B1<=0.51783).tocpu(), A1<=0.51783, "scalar gpu <= != np <=")
    
def test_greater_equal():
    A1 = np.float32(np.random.rand(10,7,10,17))
    A2 = np.float32(np.random.rand(10,7,10,17))
    v = np.float32(np.random.rand(17))
    B1 = gpu.array(A1)
    B2 = gpu.array(A2)
    w = gpu.array(v)
    
    C1 = gpu.greater_equal(B1,B1).tocpu()
    C2 = gpu.greater_equal(B1,B2).tocpu()
    C3 = gpu.greater_equal(B1,w).tocpu()
    C4 = gpu.greater_equal(B1,0.51783).tocpu()
    
    t.assert_array_equal(C1, np.greater_equal(A1,A1), "gpu.greater_equal != np.greater_equal")
    t.assert_array_equal(C2, np.greater_equal(A1,A2), "gpu.greater_equal != np.greater_equal")
    t.assert_array_equal((B1>=B1).tocpu(), A1>=A1, "gpu >= != np >=")
    t.assert_array_equal((B1>=B2).tocpu(), A1>=A2, "gpu >= != np >=") 
    
    t.assert_array_equal(C3, np.greater_equal(A1,v), "vector gpu.greater_equal != np.greater_equal")
    t.assert_array_equal((B1>=w).tocpu(), A1>=v, "vector gpu >= != np >=")
    
    t.assert_array_equal(C4, np.greater_equal(A1,0.51783), "scalar gpu.greater_equal != np.greater_equal")
    t.assert_array_equal((B1>=0.51783).tocpu(), A1>=0.51783, "scalar gpu >= != np >=")
    
def test_not_equal():
    A1 = np.float32(np.random.rand(10,7,10,17))
    A2 = np.float32(np.random.rand(10,7,10,17))
    v = np.float32(np.random.rand(17))
    B1 = gpu.array(A1)
    B2 = gpu.array(A2)
    w = gpu.array(v)
    
    C1 = gpu.not_equal(B1,B1).tocpu()
    C2 = gpu.not_equal(B1,B2).tocpu()
    C3 = gpu.not_equal(B1,w).tocpu()
    C4 = gpu.not_equal(B1,0.51783).tocpu()
    
    t.assert_array_equal(C1, np.not_equal(A1,A1), "gpu.not_equal != np.not_equal")
    t.assert_array_equal(C2, np.not_equal(A1,A2), "gpu.not_equal != np.not_equal")
    t.assert_array_equal((B1!=B1).tocpu(), A1!=A1, "gpu != != np !=")
    t.assert_array_equal((B1!=B2).tocpu(), A1!=A2, "gpu != != np !=")  
    
    t.assert_array_equal(C3, np.not_equal(A1,v), "vector gpu.not_equal != np.not_equal")
    t.assert_array_equal((B1!=w).tocpu(), A1!=v, "vector gpu != != np !=")     
    
    t.assert_array_equal(C4, np.not_equal(A1,0.51783), "scalar gpu.not_equal != np.not_equal")
    t.assert_array_equal((B1!=0.51783).tocpu(), A1!=0.51783, "scalar gpu != != np !=")
    
   
    
    
def test_slicing():
    A = np.float32(np.random.rand(17))
    B = gpu.array(A)
    C = B[-10:-2].tocpu()    
    t.assert_array_equal(C, A[-10:-2], "np[-10:-2] != gpu[-10:-2]")   
    C = B[:-2].tocpu()    
    t.assert_array_equal(C, A[:-2], "np[:-2] != gpu[:-2]")   
    C = B[-10:].tocpu()    
    t.assert_array_equal(C, A[-10:], "np[-10:] != gpu[-10:]")  
    C = B[5:-2].tocpu()    
    t.assert_array_equal(C, A[5:-2], "np[5:-2] != gpu[5:-2]")  
    C = B[-10:15].tocpu()    
    t.assert_array_equal(C, A[-10:15], "np[-10:15] != gpu[-10:15]")  
    C = B[10:15].tocpu()    
    t.assert_array_equal(C, A[10:15], "np[10:15] != gpu[10:15]")  
    C = B[10:].tocpu()    
    t.assert_array_equal(C, A[10:], "np[10:] != gpu[10:]")  
    C = B[:15].tocpu()    
    t.assert_array_equal(C, A[:15], "np[:15] != gpu[:15]")  
        
    A = np.float32(np.random.rand(17,15))
    B = gpu.array(A)
    C = B[-10:-2].tocpu()    
    t.assert_array_equal(C, A[-10:-2], "np[-10:-2] != gpu[-10:-2]")   
    C = B[:-2].tocpu()    
    t.assert_array_equal(C, A[:-2], "np[:-2] != gpu[:-2]")   
    C = B[-10:].tocpu()    
    t.assert_array_equal(C, A[-10:], "np[-10:] != gpu[-10:]")  
    C = B[5:-2].tocpu()    
    t.assert_array_equal(C, A[5:-2], "np[5:-2] != gpu[5:-2]")  
    C = B[-10:15].tocpu()    
    t.assert_array_equal(C, A[-10:15], "np[-10:15] != gpu[-10:15]")  
    C = B[10:15].tocpu()    
    t.assert_array_equal(C, A[10:15], "np[10:15] != gpu[10:15]")  
    C = B[10:].tocpu()    
    t.assert_array_equal(C, A[10:], "np[10:] != gpu[10:]")  
    C = B[:15].tocpu()    
    t.assert_array_equal(C, A[:15], "np[:15] != gpu[:15]") 
    
    C = B[-10:-2,-10:-2].tocpu()    
    t.assert_array_equal(C, A[-10:-2,-10:-2], "np[-10:-2,-10:-2] != gpu[-10:-2,-10:-2]")   
    C = B[:-2,:-2].tocpu()    
    t.assert_array_equal(C, A[:-2,:-2], "np[:-2,:-2] != gpu[:-2,:-2]")   
    C = B[-10:,-10:].tocpu()    
    t.assert_array_equal(C, A[-10:,-10:], "np[-10:,-10:] != gpu[-10:,-10:]")  
    C = B[5:-2,5:-2].tocpu()    
    t.assert_array_equal(C, A[5:-2,5:-2], "np[5:-2,5:-2] != gpu[5:-2,5:-2]")  
    C = B[-10:15,-10:15].tocpu()    
    t.assert_array_equal(C, A[-10:15,-10:15], "np[-10:15,-10:15] != gpu[-10:15,-10:15]")  
    C = B[10:15,10:15].tocpu()    
    t.assert_array_equal(C, A[10:15,10:15], "np[10:15,10:15] != gpu[10:15,10:15]")  
    C = B[10:,10:].tocpu()    
    t.assert_array_equal(C, A[10:,10:], "np[10:,10:] != gpu[10:,10:]")  
    C = B[:15,:15].tocpu()    
    t.assert_array_equal(C, A[:15,:15], "np[:15,:15] != gpu[:15,:15]")    
    
    A = np.float32(np.random.rand(17,15,23))
    B = gpu.array(A)
    C = B[-10:-2].tocpu()    
    t.assert_array_equal(C, A[-10:-2], "np[-10:-2] != gpu[-10:-2]")   
    C = B[:-2].tocpu()    
    t.assert_array_equal(C, A[:-2], "np[:-2] != gpu[:-2]")   
    C = B[-10:].tocpu()    
    t.assert_array_equal(C, A[-10:], "np[-10:] != gpu[-10:]")  
    C = B[5:-2].tocpu()    
    t.assert_array_equal(C, A[5:-2], "np[5:-2] != gpu[5:-2]")  
    C = B[-10:15].tocpu()    
    t.assert_array_equal(C, A[-10:15], "np[-10:15] != gpu[-10:15]")  
    C = B[10:15].tocpu()    
    t.assert_array_equal(C, A[10:15], "np[10:15] != gpu[10:15]")  
    C = B[10:].tocpu()    
    t.assert_array_equal(C, A[10:], "np[10:] != gpu[10:]")  
    C = B[:15].tocpu()    
    t.assert_array_equal(C, A[:15], "np[:15] != gpu[:15]") 
    
    C = B[-10:-2,-10:-2].tocpu()    
    t.assert_array_equal(C, A[-10:-2,-10:-2], "np[-10:-2,-10:-2] != gpu[-10:-2,-10:-2]")   
    C = B[:-2,:-2].tocpu()    
    t.assert_array_equal(C, A[:-2,:-2], "np[:-2,:-2] != gpu[:-2,:-2]")   
    C = B[-10:,-10:].tocpu()    
    t.assert_array_equal(C, A[-10:,-10:], "np[-10:,-10:] != gpu[-10:,-10:]")  
    C = B[5:-2,5:-2].tocpu()    
    t.assert_array_equal(C, A[5:-2,5:-2], "np[5:-2,5:-2] != gpu[5:-2,5:-2]")  
    C = B[-10:15,-10:15].tocpu()    
    t.assert_array_equal(C, A[-10:15,-10:15], "np[-10:15,-10:15] != gpu[-10:15,-10:15]")  
    C = B[10:15,10:15].tocpu()    
    t.assert_array_equal(C, A[10:15,10:15], "np[10:15,10:15] != gpu[10:15,10:15]")  
    C = B[10:,10:].tocpu()    
    t.assert_array_equal(C, A[10:,10:], "np[10:,10:] != gpu[10:,10:]")  
    C = B[:15,:15].tocpu()    
    t.assert_array_equal(C, A[:15,:15], "np[:15,:15] != gpu[:15,:15]")   
    
    C = B[-10:-2,-10:-2,-10:-2].tocpu()    
    t.assert_array_equal(C, A[-10:-2,-10:-2,-10:-2], "np[-10:-2,-10:-2,-10:-2] != gpu[-10:-2,-10:-2,-10:-2]")   
    C = B[:-2,:-2,:-2].tocpu()    
    t.assert_array_equal(C, A[:-2,:-2,:-2], "np[:-2,:-2,:-2] != gpu[:-2,:-2,:-2]")     
    C = B[-10:,-10:,-10:].tocpu()    
    t.assert_array_equal(C, A[-10:,-10:,-10:], "np[-10:,-10:,-10:] != gpu[-10:,-10:,-10:]")  
    C = B[5:-2,5:-2,5:-2].tocpu()    
    t.assert_array_equal(C, A[5:-2,5:-2,5:-2], "np[5:-2,5:-2,5:-2] != gpu[5:-2,5:-2,5:-2]")  
    C = B[-10:15,-10:15,-10:15].tocpu()    
    t.assert_array_equal(C, A[-10:15,-10:15,-10:15], "np[-10:15,-10:15,-10:15] != gpu[-10:15,-10:15,-10:15]")  
    C = B[10:15,10:15,10:15].tocpu()    
    t.assert_array_equal(C, A[10:15,10:15,10:15], "np[10:15,10:15,10:15] != gpu[10:15,10:15,10:15]")  
    C = B[10:,10:,10:].tocpu()    
    t.assert_array_equal(C, A[10:,10:,10:], "np[10:,10:,10:] != gpu[10:,10:,10:]")  
    C = B[:15,:15,:15].tocpu()    
    t.assert_array_equal(C, A[:15,:15,:15], "np[:15,:15,:15] != gpu[:15,:15,:15]")  
    
    A = np.float32(np.random.rand(17,15,23,21))
    B = gpu.array(A)
    C = B[-10:-2].tocpu()    
    t.assert_array_equal(C, A[-10:-2], "np[-10:-2] != gpu[-10:-2]")   
    C = B[:-2].tocpu()    
    t.assert_array_equal(C, A[:-2], "np[:-2] != gpu[:-2]")   
    C = B[-10:].tocpu()    
    t.assert_array_equal(C, A[-10:], "np[-10:] != gpu[-10:]")  
    C = B[5:-2].tocpu()    
    t.assert_array_equal(C, A[5:-2], "np[5:-2] != gpu[5:-2]")  
    C = B[-10:15].tocpu()    
    t.assert_array_equal(C, A[-10:15], "np[-10:15] != gpu[-10:15]")  
    C = B[10:15].tocpu()    
    t.assert_array_equal(C, A[10:15], "np[10:15] != gpu[10:15]")  
    C = B[10:].tocpu()    
    t.assert_array_equal(C, A[10:], "np[10:] != gpu[10:]")  
    C = B[:15].tocpu()    
    t.assert_array_equal(C, A[:15], "np[:15] != gpu[:15]") 
    
    C = B[-10:-2,-10:-2].tocpu()    
    t.assert_array_equal(C, A[-10:-2,-10:-2], "np[-10:-2,-10:-2] != gpu[-10:-2,-10:-2]")   
    C = B[:-2,:-2].tocpu()    
    t.assert_array_equal(C, A[:-2,:-2], "np[:-2,:-2] != gpu[:-2,:-2]")   
    C = B[-10:,-10:].tocpu()    
    t.assert_array_equal(C, A[-10:,-10:], "np[-10:,-10:] != gpu[-10:,-10:]")  
    C = B[5:-2,5:-2].tocpu()    
    t.assert_array_equal(C, A[5:-2,5:-2], "np[5:-2,5:-2] != gpu[5:-2,5:-2]")  
    C = B[-10:15,-10:15].tocpu()    
    t.assert_array_equal(C, A[-10:15,-10:15], "np[-10:15,-10:15] != gpu[-10:15,-10:15]")  
    C = B[10:15,10:15].tocpu()    
    t.assert_array_equal(C, A[10:15,10:15], "np[10:15,10:15] != gpu[10:15,10:15]")  
    C = B[10:,10:].tocpu()    
    t.assert_array_equal(C, A[10:,10:], "np[10:,10:] != gpu[10:,10:]")  
    C = B[:15,:15].tocpu()    
    t.assert_array_equal(C, A[:15,:15], "np[:15,:15] != gpu[:15,:15]")   
    
    C = B[-10:-2,-10:-2,-10:-2].tocpu()    
    t.assert_array_equal(C, A[-10:-2,-10:-2,-10:-2], "np[-10:-2,-10:-2,-10:-2] != gpu[-10:-2,-10:-2,-10:-2]")   
    C = B[:-2,:-2,:-2].tocpu()    
    t.assert_array_equal(C, A[:-2,:-2,:-2], "np[:-2,:-2,:-2] != gpu[:-2,:-2,:-2]")     
    C = B[-10:,-10:,-10:].tocpu()    
    t.assert_array_equal(C, A[-10:,-10:,-10:], "np[-10:,-10:,-10:] != gpu[-10:,-10:,-10:]")  
    C = B[5:-2,5:-2,5:-2].tocpu()    
    t.assert_array_equal(C, A[5:-2,5:-2,5:-2], "np[5:-2,5:-2,5:-2] != gpu[5:-2,5:-2,5:-2]")  
    C = B[-10:15,-10:15,-10:15].tocpu()    
    t.assert_array_equal(C, A[-10:15,-10:15,-10:15], "np[-10:15,-10:15,-10:15] != gpu[-10:15,-10:15,-10:15]")  
    C = B[10:15,10:15,10:15].tocpu()    
    t.assert_array_equal(C, A[10:15,10:15,10:15], "np[10:15,10:15,10:15] != gpu[10:15,10:15,10:15]")  
    C = B[10:,10:,10:].tocpu()    
    t.assert_array_equal(C, A[10:,10:,10:], "np[10:,10:,10:] != gpu[10:,10:,10:]")  
    C = B[:15,:15,:15].tocpu()    
    t.assert_array_equal(C, A[:15,:15,:15], "np[:15,:15,:15] != gpu[:15,:15,:15]") 
    
    C = B[-10:-2,-10:-2,-10:-2,-10:-2].tocpu()    
    t.assert_array_equal(C, A[-10:-2,-10:-2,-10:-2,-10:-2], "np[-10:-2,-10:-2,-10:-2,-10:-2] != gpu[-10:-2,-10:-2,-10:-2,-10:-2]")   
    C = B[:-2,:-2,:-2,:-2].tocpu()    
    t.assert_array_equal(C, A[:-2,:-2,:-2,:-2], "np[:-2,:-2,:-2,:-2] != gpu[:-2,:-2,:-2,:-2]")     
    C = B[-10:,-10:,-10:,-10:].tocpu()    
    t.assert_array_equal(C, A[-10:,-10:,-10:,-10:], "np[-10:,-10:,-10:,-10:] != gpu[-10:,-10:,-10:,-10:]")  
    C = B[5:-2,5:-2,5:-2,5:-2].tocpu()    
    t.assert_array_equal(C, A[5:-2,5:-2,5:-2,5:-2], "np[5:-2,5:-2,5:-2,5:-2] != gpu[5:-2,5:-2,5:-2,5:-2]")  
    C = B[-10:15,-10:15,-10:15,-10:15].tocpu()    
    t.assert_array_equal(C, A[-10:15,-10:15,-10:15,-10:15], "np[-10:15,-10:15,-10:15,-10:15] != gpu[-10:15,-10:15,-10:15,-10:15]")  
    C = B[10:15,10:15,10:15,10:15].tocpu()    
    t.assert_array_equal(C, A[10:15,10:15,10:15,10:15], "np[10:15,10:15,10:15,10:15] != gpu[10:15,10:15,10:15,10:15]")  
    C = B[10:,10:,10:,10:].tocpu()    
    t.assert_array_equal(C, A[10:,10:,10:,10:], "np[10:,10:,10:,10:] != gpu[10:,10:,10:,10:]")  
    C = B[:15,:15,:15,:15].tocpu()    
    t.assert_array_equal(C, A[:15,:15,:15,:15], "np[:15,:15,:15,:15] != gpu[:15,:15,:15,:15]")   
    
def test_dot():
    A1 = np.float32(np.random.rand(17,83))
    B1 = np.float32(np.random.rand(83,13)) 
    A2 = gpu.array(A1)
    B2 = gpu.array(B1)
    C = gpu.dot(A2,B2)
    t.assert_array_almost_equal(C.tocpu(), np.dot(A1,B1), 5, "np.dot != gpu.dot 2 dimensions!")
    
    C*=0.0
    gpu.dot(A2,B2,C)
    t.assert_array_almost_equal(C.tocpu(), np.dot(A1,B1), 5, "np.dot != gpu.dot 2 dimensions!")
    
    
    A3 = np.float32(np.random.rand(83,17))
    B3 = np.float32(np.random.rand(13,83))
    A4 = gpu.array(A3) 
    B4 = gpu.array(B3)
    
    gpu.dotT(A2,B4,C)
    t.assert_array_almost_equal(C.tocpu(), np.dot(A1,B3.T), 5, "np.dotT != gpu.dot 2 dimensions!")  
    C*=0.0  
    gpu.dotT(A2,B4,C)
    t.assert_array_almost_equal(C.tocpu(), np.dot(A1,B3.T), 5, "np.dotT != gpu.dot 2 dimensions!")
    
    gpu.Tdot(A4,B2,C)
    t.assert_array_almost_equal(C.tocpu(), np.dot(A3.T,B1), 5, "np.Tdot != gpu.dot 2 dimensions!")  
    C*=0.0  
    gpu.Tdot(A4,B2,C)
    t.assert_array_almost_equal(C.tocpu(), np.dot(A3.T,B1), 5, "np.Tdot != gpu.dot 2 dimensions!")
    

def test_synchronizingAdd():
    A = np.float32(np.random.rand(17,83))
    B = gpu.array(A)        
    gpu.enable_peer_access()
    C = gpu.synchronizingAdd(B)   
    
    t.assert_array_almost_equal(C.tocpu(), A*gpu.gpu_count(), 7, "Synchronizing add does not work!")
    C*=0
    gpu.synchronizingAdd(B,C)
    gpu.disable_peer_access()
    t.assert_array_almost_equal(C.tocpu(), A*gpu.gpu_count(), 7, "Synchronizing add does not work!")

        
def test_allocator_init():    
    data = np.float32(np.random.rand(5333,256))
    labels = np.float32(np.random.randint(0,10,(5333,)))
    #labels = np.float32(np.random.rand(5333,4))
    
    batch_size = 128
    alloc = batch_allocator(data, labels, 0.3, 0.3, batch_size)
    for epoch in range(10):
        print 'EPOCH: {0}'.format(epoch+1)
        for batchno, i in enumerate(alloc.train()):
            stop_idx = (np.int32(np.round(data.shape[0]*0.4)) if i+batch_size > np.int32(np.round(data.shape[0]*0.4)) else i+batch_size)
            batch = data[i:stop_idx]
            batch_y = u.create_t_matrix(labels[i:stop_idx],10)
            #batch_y = labels[i:stop_idx]
            t.assert_equal(alloc.batch.tocpu(), batch)
            t.assert_equal(alloc.batch_y.tocpu(),batch_y )
            
        t.assert_equal(batchno+1, alloc.batch_count[0])
        
        
    for epoch in range(10): 
        for batchno, i in enumerate(alloc.cv()):                 
            stop_idx = (np.int32(np.round(data.shape[0]*0.7)) if i+batch_size > np.int32(np.round(data.shape[0]*0.7)) else i+batch_size)
            batch = data[i:stop_idx]
            batch_y = u.create_t_matrix(labels[i:stop_idx],10)
            #batch_y = labels[i:stop_idx]
            C1 =  alloc.batch.tocpu()
            C2 =  alloc.batch_y.tocpu()
            if len(C1.shape) == 1: 
                C1 = C1[np.newaxis,:]
                C2 = C2[np.newaxis,:]
            t.assert_equal(C1, batch)
            t.assert_equal(C2,batch_y )
        t.assert_equal(batchno+1, alloc.batch_count[1])
            
    for epoch in range(10):             
        for i in alloc.test():                 
            stop_idx = (np.int32(np.round(data.shape[0]*1.0)) if i+batch_size > np.int32(np.round(data.shape[0]*1.0)) else i+batch_size)
            batch = data[i:stop_idx]
            batch_y = u.create_t_matrix(labels[i:stop_idx],10)
            #batch_y = labels[i:stop_idx]
            C1 =  alloc.batch.tocpu()
            C2 =  alloc.batch_y.tocpu()
            if len(C1.shape) == 1: 
                C1 = C1[np.newaxis,:]
                C2 = C2[np.newaxis,:]
            t.assert_equal(C1, batch)
            t.assert_equal(C2,batch_y )
            
    for epoch in range(10):             
        for batchno, i in enumerate(alloc.test()):                    
            stop_idx = (np.int32(np.round(data.shape[0]*1.0)) if i+batch_size > np.int32(np.round(data.shape[0]*1.0)) else i+batch_size)
            batch = data[i:stop_idx]
            batch_y = u.create_t_matrix(labels[i:stop_idx],10)
            #batch_y = labels[i:stop_idx]
            C1 =  alloc.batch.tocpu()
            C2 =  alloc.batch_y.tocpu()
            if len(C1.shape) == 1: 
                C1 = C1[np.newaxis,:]
                C2 = C2[np.newaxis,:]
            t.assert_equal(C1, batch)
            t.assert_equal(C2,batch_y )
        t.assert_equal(batchno+1, alloc.batch_count[2])
            
                
        for i in alloc.train():                 
            stop_idx = (np.int32(np.round(data.shape[0]*0.4)) if i+batch_size > np.int32(np.round(data.shape[0]*0.4)) else i+batch_size)
            batch = data[i:stop_idx]
            batch_y = u.create_t_matrix(labels[i:stop_idx],10)
            #batch_y = labels[i:stop_idx]
            C1 =  alloc.batch.tocpu()
            C2 =  alloc.batch_y.tocpu()
            if len(C1.shape) == 1: 
                C1 = C1[np.newaxis,:]
                C2 = C2[np.newaxis,:]
            t.assert_equal(C1, batch)
            t.assert_equal(C2,batch_y )
            
            
        for i in alloc.train():                 
            stop_idx = (np.int32(np.round(data.shape[0]*0.4)) if i+batch_size > np.int32(np.round(data.shape[0]*0.4)) else i+batch_size)
            batch = data[i:stop_idx]
            batch_y = u.create_t_matrix(labels[i:stop_idx],10)
            #batch_y = labels[i:stop_idx]
            C1 =  alloc.batch.tocpu()
            C2 =  alloc.batch_y.tocpu()
            if len(C1.shape) == 1: 
                C1 = C1[np.newaxis,:]
                C2 = C2[np.newaxis,:]
            t.assert_equal(C1, batch)
            t.assert_equal(C2,batch_y )
            
            
        for i in alloc.cv():                 
            stop_idx = (np.int32(np.round(data.shape[0]*0.7)) if i+batch_size > np.int32(np.round(data.shape[0]*0.7)) else i+batch_size)
            batch = data[i:stop_idx]
            batch_y = u.create_t_matrix(labels[i:stop_idx],10)
            #batch_y = labels[i:stop_idx]
            C1 =  alloc.batch.tocpu()
            C2 =  alloc.batch_y.tocpu()
            if len(C1.shape) == 1: 
                C1 = C1[np.newaxis,:]
                C2 = C2[np.newaxis,:]
            t.assert_equal(C1, batch)
            t.assert_equal(C2,batch_y )
            
            
        for i in alloc.train():                 
            stop_idx = (np.int32(np.round(data.shape[0]*0.4)) if i+batch_size > np.int32(np.round(data.shape[0]*0.4)) else i+batch_size)
            batch = data[i:stop_idx]
            batch_y = u.create_t_matrix(labels[i:stop_idx],10)
            #batch_y = labels[i:stop_idx]
            C1 =  alloc.batch.tocpu()
            C2 =  alloc.batch_y.tocpu()
            if len(C1.shape) == 1: 
                C1 = C1[np.newaxis,:]
                C2 = C2[np.newaxis,:]
            t.assert_equal(C1, batch)
            t.assert_equal(C2,batch_y )
            
            
        for i in alloc.test():                 
            stop_idx = (np.int32(np.round(data.shape[0]*1.0)) if i+batch_size > np.int32(np.round(data.shape[0]*1.0)) else i+batch_size)
            batch = data[i:stop_idx]
            batch_y = u.create_t_matrix(labels[i:stop_idx],10)
            #batch_y = labels[i:stop_idx]
            C1 =  alloc.batch.tocpu()
            C2 =  alloc.batch_y.tocpu()
            if len(C1.shape) == 1: 
                C1 = C1[np.newaxis,:]
                C2 = C2[np.newaxis,:]
            t.assert_equal(C1, batch)
            t.assert_equal(C2,batch_y )
    
        
    t0 = time.time()
    
    for epoch in range(10):
        for i in alloc.train():
            pass        
            
    sec = time.time()-t0
    GB = 10*data.shape[0]*data.shape[1]*4*(1024**-3)
    
    assert GB/sec > 1.75
    #TODO: this should be closer to 8GB/s -> use pinned memory 
   
    
def test_dropout():
    A = np.float32(np.random.rand(14,13,17,83))
    B = gpu.array(A)
    C = gpu.dropout(B, 0.2)
    
    t.assert_almost_equal(C.tocpu().sum()/100000, A.sum()*0.8/100000, 2, "Dropout seems to be fishy")
    
    C = gpu.dropout(B, 0.5)  
    t.assert_almost_equal(C.tocpu().sum()/100000, A.sum()*0.5/100000, 2, "Dropout seems to be fishy")
    
    C = gpu.dropout(B, 0.8)   
    t.assert_almost_equal(C.tocpu().sum()/100000, A.sum()*0.2/100000, 2, "Dropout seems to be fishy")    
    
    C = gpu.dropout(B, 1.0)    
    t.assert_almost_equal(C.tocpu().sum(), 0.0, 0, "Dropout seems to be fishy")
    
    C*=0
    gpu.dropout(B,0.2,C)
    t.assert_almost_equal(C.tocpu().sum()/100000, A.sum()*0.8/100000, 2, "Dropout seems to be fishy")    
    
    C*=0
    gpu.dropout(B,0.5,C)
    t.assert_almost_equal(C.tocpu().sum()/100000, A.sum()*0.5/100000, 2, "Dropout seems to be fishy")    
    
    C*=0
    gpu.dropout(B,0.8,C)
    t.assert_almost_equal(C.tocpu().sum()/100000, A.sum()*0.2/100000, 2, "Dropout seems to be fishy")    
    
    C*=0
    gpu.dropout(B,1.0,C)
    t.assert_almost_equal(C.tocpu().sum(), 0.0, 0, "Dropout seems to be fishy")
    
def test_rectified_linear():
    A = np.float32(np.random.rand(2,2,17,83))
    B = gpu.array(A)
    C = gpu.ReLU(B)
    t.assert_equal(C.tocpu(), A*(A>0), "Bad rectified linear values")    
    C = gpu.ReLU_grad(B)
    t.assert_equal(C.tocpu(), A>0, "Bad rectified linear grad values")
    C*=0
    gpu.ReLU(B,C)
    t.assert_equal(C.tocpu(), A*(A>0), "Bad rectified linear values")
    C*=0    
    gpu.ReLU_grad(B,C)
    t.assert_equal(C.tocpu(), A>0, "Bad rectified linear grad values")
    
def test_softmax():
    A = np.float32(np.random.rand(17,83))
    B = gpu.array(A)
    C = gpu.softmax(B)
    
    t.assert_array_almost_equal(C.tocpu(), u.softmax(A), 4, "Softmax problem!")
    C*=0
    gpu.softmax(B,C)
    t.assert_array_almost_equal(C.tocpu(), u.softmax(A), 4, "Softmax problem!")
    
def test_argmax():
    A = np.float32(np.random.rand(17,83))
    B = gpu.array(A)
    C = gpu.argmax(B)
    
    t.assert_array_almost_equal(C.tocpu(), np.argmax(A,1)[np.newaxis].T, 4, "Softmax problem!")
    C*=0
    gpu.argmax(B,C)
    t.assert_array_almost_equal(C.tocpu(), np.argmax(A,1)[np.newaxis].T, 4, "Softmax problem!")
    
def test_reduce_functions():
    A = np.float32(np.random.randn(2,3,17,83))
    B = gpu.array(A)
    t.assert_equal(B.max(), A.max(), "max not equal")
    t.assert_equal(B.min(), A.min(), "min not equal")
    t.assert_array_almost_equal(B.sum(), A.sum(), 3, "sum not equal")
    
def test_linear():
    A = np.float32(np.random.rand(17,83))
    B = gpu.array(A)
    C = gpu.linear(B)
    
    t.assert_array_equal(C.tocpu(), A, "Copy/linear not working!")
    C*=0
    gpu.linear(B,C)
    t.assert_array_equal(C.tocpu(), A, "Copy/linear not working!")


def test_layer():
    net = Layer()
    net.add(Layer(800, Logistic()))
    net.add(Layer(10,Softmax()))
    
    X = np.load('./mnist_mini_X.npy')
    y = np.load('./mnist_mini_y.npy')
    #X = np.load('/home/tim/data/MNIST/train_X.npy')
    #y = np.load('/home/tim/data/MNIST/train_y.npy')
    
    alloc = batch_allocator(X,y, 0.2,0.0,32)   
    net.set_config_value('dropout', 0.5)
    net.set_config_value('input_dropout', 0.2) 
    for epoch in range(15):
        t0 = time.time()    
        for i in alloc.train():   
            #net.forward(gpu.array(batch),gpu.array(batch_y))
            
            net.forward(alloc.batch,alloc.batch_y)
            net.backward_errors()
            net.backward_grads()
            net.accumulate_error()
            net.weight_update()
        net.print_reset_error()
        #print 'train epoch time: {0} secs'.format(time.time()-t0)
        
        t0 = time.time()    
        for i in alloc.cv():
            net.forward(alloc.batch,alloc.batch_y)
            net.accumulate_error()
        net.print_reset_error('Cv')
        #print 'cv error time: {0} secs'.format(time.time()-t0)
            
        #print net.w_next.tocpu().sum()
        #print np.sum((C2-y)**2)
    
    #print np.sum((C2-y)**2)
    #print C2[0:20].T
    #print y[0:20].T
    
    C2 = net.predict(gpu.array(X)).tocpu()
    print np.sum((C2-y)**2)    
    assert np.sum((C2-y)**2) < 500


def split_add_test():
    gpu.enable_peer_access()
    for i in range(500):
        dims = np.random.randint(2,5,(2,))
        A1 = np.random.rand(dims[0],dims[1])
        A2 = np.random.rand(dims[0],dims[1])
        C1 = gpu.empty((dims[1],dims[1]))
        C2 = gpu.zeros((dims[1],dims[1]))                
        B1 = gpu.array(A1,split_idx=2)
        B2 = gpu.array(A2,split_idx=2)
        gpu.Tdot(B2,B1,C1)      
        gpu.synchronizingAdd(C1,C2)        
        
        C = np.dot(A2.T,A1)
        D = np.ones_like(C)*0.05
        print i
        print dims
        #the dot product is just inherently unstable
        print np.max(((C-C2.tocpu())**2)/C.size)
        #print C
        #print C2.tocpu()
        t.assert_array_less(((C-C2.tocpu())**2)/C.size, D, 'split add dot product chain yields wrong result!')    
        #print [C2.tocpu().sum()/10000,C.sum()/10000]


def test_slice_or_stack_axis():
    for i in range(500):
        dims = np.random.randint(5,50,(2,))
        A = np.random.rand(dims[0],dims[1])
        B1 = gpu.array(A)
        B2 = gpu.zeros((dims[0],dims[1]),2)
        gpu.slice_or_stack_axis(B1, B2)
        
        C = gpu.zeros((dims[0],dims[1]))
        gpu.slice_or_stack_axis(B2, C)
        t.assert_array_almost_equal(C.tocpu(), A, 3, "slice and stack row not working!")
    gpu.disable_peer_access()
    #assert False
    
  
    

    
    
if __name__ == '__main__':    
    nose.run()