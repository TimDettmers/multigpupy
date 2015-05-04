
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
from neural_net import Neural_net

def setup():
    pass

def teardown():
    pass

def test_cpu():
    A = np.float32(np.random.rand(17,83))
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
    
    A = np.float32(np.random.rand(3,3,3))
    C1 = gpu.array(A).T.tocpu()
    C2 = np.transpose(A,(0,2,1))
    print C1
    print C2
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
    
def test_double_ReLU():    
    A = np.float32(np.random.randn(10,7,83,4))
    B = gpu.array(A)
    C = gpu.double_ReLU(B).tocpu()  
    t.assert_array_almost_equal(C, A*(A>0.0)*(A<1.0), 5, "Double relu not like numpy equivalent")
    
    
def test_double_ReLU_grad():    
    A = np.float32(np.random.randn(10,7,83,4))
    C2 = np.ones_like(A)
    B = gpu.array(A)
    C = gpu.double_ReLU_grad(B).tocpu()  
    t.assert_array_almost_equal(C, C2*(A>0.0)*(A<1.0), 5, "Double relu grad not like numpy equivalent")
    
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
    A2 = np.float32(np.random.rand(10,7,83,4))
    B = gpu.array(A)
    B2 = gpu.array(A2)
    C = gpu.power(B,5).tocpu()  
    t.assert_array_almost_equal(C, np.power(A,5), 5, "power not like numpy equivalent") 
    C = gpu.power(B,17.83).tocpu()  
    t.assert_array_almost_equal(C, np.power(A,17.83), 5, "power not like numpy equivalent") 
    
    
    C = (B**5).tocpu()  
    t.assert_array_almost_equal(C, np.power(A,5), 5, "power not like numpy equivalent") 
    C = (B**17.83).tocpu()  
    t.assert_array_almost_equal(C, np.power(A,17.83), 5, "power not like numpy equivalent") 
    
    C = gpu.power(B-B2,5).tocpu()  
    t.assert_array_almost_equal(C, np.power(A-A2,5), 5, "power not like numpy equivalent") 
    C = gpu.power(B-B2,2).tocpu()  
    t.assert_array_almost_equal(C, np.power(A-A2,2), 5, "power not like numpy equivalent") 
    
    C = ((B-B2)**5).tocpu()  
    t.assert_array_almost_equal(C, np.power(A-A2,5), 5, "power not like numpy equivalent") 
    C = ((B-B2)**2).tocpu()  
    t.assert_array_almost_equal(C, np.power(A-A2,2), 5, "power not like numpy equivalent") 
    
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
    for i in range(100):        
        dims = np.random.randint(5,250,(3,))
        A1 = np.float32(np.random.rand(dims[0],dims[1]))
        B1 = np.float32(np.random.rand(dims[1],dims[2])) 
        A2 = gpu.array(A1)
        B2 = gpu.array(B1)
        C = gpu.dot(A2,B2)
        t.assert_array_almost_equal(C.tocpu(), np.dot(A1,B1), 3, "np.dot != gpu.dot 2 dimensions!")
        
        C*=0.0
        gpu.dot(A2,B2,C)
        t.assert_array_almost_equal(C.tocpu(), np.dot(A1,B1), 3, "np.dot != gpu.dot 2 dimensions!")
        
        
        A3 = np.float32(np.random.rand(dims[1],dims[0]))
        B3 = np.float32(np.random.rand(dims[2],dims[1]))
        A4 = gpu.array(A3) 
        B4 = gpu.array(B3)
        
        gpu.dotT(A2,B4,C)
        t.assert_array_almost_equal(C.tocpu(), np.dot(A1,B3.T), 3, "np.dotT != gpu.dot 2 dimensions!")  
        C*=0.0  
        gpu.dotT(A2,B4,C)
        t.assert_array_almost_equal(C.tocpu(), np.dot(A1,B3.T), 3, "np.dotT != gpu.dot 2 dimensions!")
        
        gpu.Tdot(A4,B2,C)
        t.assert_array_almost_equal(C.tocpu(), np.dot(A3.T,B1), 3, "np.Tdot != gpu.dot 2 dimensions!")  
        C*=0.0  
        gpu.Tdot(A4,B2,C)
        t.assert_array_almost_equal(C.tocpu(), np.dot(A3.T,B1), 3, "np.Tdot != gpu.dot 2 dimensions!")
    
        
def test_batch_allocator_sequential():    
    data = np.float32(np.random.rand(5000,784))
    labels = np.float32(np.random.randint(0,10,(5000,)))
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
            #print data[0:3]
            #gpu.print_tensor(alloc.batch)
            #print i
            #print batch_y, alloc.batch_y.tocpu()
            #print alloc.batch.shape
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
    print GB/sec
    assert GB/sec > 4.5    
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
    net.set_config_value('dropout', 0.0)
    net.set_config_value('input_dropout', 0.0) 
    net.set_config_value('parallelism','None')
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
    
    C2 = net.predict(gpu.array(X[0:X.shape[0]*0.8])).tocpu()
    print np.sum((C2-y[0:X.shape[0]*0.8])**2)    
    assert np.sum((C2-y[0:X.shape[0]*0.8])**2) < 50

    
def test_sync():
    gpu.enable_peer_access()
    #logistic_func = lambda x: 1.0/(1.0+np.exp(-x))
    for i in range(1):
        dims = np.random.randint(5,100,(2,))        
        dims[0]+=  gpu.gpu_count() - (dims[0] % gpu.gpu_count())
        A1 = np.float32(np.random.rand(dims[0],dims[1]))
        w = np.float32(np.random.rand(dims[1],10))
        W1 = gpu.array(w)
        B1 = gpu.array(A1,split_idx=2)
        B2 = gpu.empty((dims[0], 10), split_idx=2)
        gpu.sync(B1)
        C1 = gpu.sync_streams_add(B1)
        dim = dims[0]/4
        C =  A1[0:dim] + A1[dim:2*dim] + A1[2*dim:3*dim] + A1[3*dim:4*dim]
        t.assert_almost_equal(C,C1.tocpu(),4,'split sync with add not working')
        
        
    dims = [5120,2560]
    A1 = np.float32(np.random.rand(dims[0],dims[1])) 
    B1 = gpu.array(A1,split_idx=2)    
    t0 = time.time()
    for i in range(10):
        gpu.sync(B1)
        gpu.sync_streams_add(B1)
    GB = 10*4*dims[0]*dims[1]*1024**-3
    secs = time.time()-t0
    print GB/secs  
    t.assert_(GB/secs > 2.5, "transfer rate below 2.5 GB/s!")
    
    


def split_add_test():
    for i in range(50):
        dims = np.random.randint(5,100,(2,))     
        dims[0]+=  gpu.gpu_count() - (dims[0] % gpu.gpu_count())    
        A1 = np.float32(np.random.rand(dims[0],dims[1]))
        A2 = np.float32(np.random.rand(dims[0],dims[1]))
        C1 = gpu.empty((dims[1],dims[1]))    
         
        C5 = gpu.empty((dims[1],dims[1]))
         
        B1 = gpu.array(A1,split_idx=2)        
        B2 = gpu.array(A2,split_idx=2)
           
        print B1.shape
        gpu.Tdot(B2,B1,C1)    
        gpu.sync(C1)          
        #the dot product is just inherently unstable
        C5 = gpu.sync_streams_add(C1,split_idx=-1)     
        C = np.dot(A2.T,A1)
        #errors = np.sqrt(((C-C3.tocpu())**2))
        #print C[0,0:5]
        #print C3.tocpu()[0,0:5]
        #print np.sort(np.sqrt(((C-C3.tocpu())**2)).flatten())[::-1][0:5]
        #print np.sqrt(((C-C3.tocpu())**2))[0,0:5]
        #print '----------'
        t.assert_array_almost_equal(C5.tocpu(), C, 4,'split add dot product chain yields wrong result!')    
        #print [C2.tocpu().sum()/10000,C.sum()/10000]
    
   

    
def test_slice_or_stack_axis():
    for i in range(50):
        dims = np.random.randint(5,50,(2,))
        dims[0]+=  gpu.gpu_count() - (dims[0] % gpu.gpu_count())   
        A = np.random.rand(dims[0],dims[1])
        B1 = gpu.array(A)
        B2 = gpu.zeros((dims[0],dims[1]),2)
        gpu.slice_axis(B1, B2)
        
        C = gpu.zeros((dims[0],dims[1]))
        gpu.stack_axis(B2, C)
        t.assert_array_almost_equal(C.tocpu(), A, 3, "slice and stack row not working!")
    #assert False
    

def test_batch_allocator_parallelism():
    net = Layer()
    #net.set_config_value('parallelism','data')
    data = np.float32(np.random.rand(5400,4))
    labels = np.float32(np.random.randint(0,10,(5400,)))
    
    batch_size = 128
    alloc = batch_allocator(data, labels, 0.3, 0.3, batch_size, 'parallel')
    alloc.net = net
    alloc.peer_access_enabled = True
    for i in alloc.train():     
        #if A.shape[2] < gpu.gpu_count() or A.shape[2] % 2 != 0: continue        
        stop_idx = (np.int32(np.round(data.shape[0]*0.4)) if i+batch_size > np.int32(np.round(data.shape[0]*0.4)) else i+batch_size)
        batch = data[i:stop_idx]
        batch_y = u.create_t_matrix(labels[i:stop_idx],10)
        C1 = gpu.empty(alloc.current.shape)        
        B2 = gpu.empty((batch.shape[1], batch.shape[1]))        
        
        gpu.stack_axis(alloc.batch, C1)    
        t.assert_array_equal(C1.tocpu(), batch, "stack allocator data parallelism not working!")
                       
        gpu.Tdot(alloc.batch,alloc.batch,B2)        
        gpu.sync(B2)
        B6 = gpu.sync_streams_add(B2,split_idx=-1)
        C2 = np.dot(batch.T,batch)  
        print i
        errors = np.sqrt((C2-B6.tocpu())**2).flatten()
        print errors
        t.assert_array_almost_equal(C2, B6.tocpu(),2, "synch add data parallelism not working!")
       
           
    
def test_arregates():
    A = np.float32(np.random.randn(4,13,8,4))
    B1 = gpu.array(A)
    B2 = gpu.array(A,split_idx=2)
    
    t.assert_equal(np.max(A),B1.max(),"Thrust max")
    t.assert_equal(np.max(A),B2.max(),"Thrust max with split")
    t.assert_equal(np.min(A),B1.min(),"Thrust min")
    t.assert_equal(np.min(A),B2.min(),"Thrust min with split")
    t.assert_almost_equal(np.sum(A),B1.sum(),3,"Thrust sum")
    t.assert_almost_equal(np.sum(A),B2.sum(),3,"Thrust sum with split")
    
  
def test_neural_net():
    net = Neural_net(epochs=15)

    X = np.load('./mnist_mini_X.npy')
    y = np.load('./mnist_mini_y.npy')
    
    net.fit(X,y,batch_size=32)    
    pred = net.predict_proba(X[X.shape[0]*0.8:])
    error = 1.0-((np.argmax(pred,1)==y[X.shape[0]*0.8:].T).sum()/(y.size*0.2))
    print error
    assert error < 0.20
   

def test_8bit_compression():
    A = np.float32(np.random.rand(500,50,30,4))
    B1 = gpu.array(A)
    C = gpu.zeros(B1.shape)
    B2 = gpu.empty_char_like(B1)
    max_value = np.max(np.abs(A))
    gpu.compress_8bit(B1, max_value, B2)
    gpu.decompress_8bit(B2, max_value, C)
    abs_error = np.sqrt((A-C.tocpu())**2)
    rel_error = np.mean(abs_error/np.abs(A))
    
    abs_error16bit = np.sqrt((A-np.float16(A))**2)
    rel_error16bit = np.mean(abs_error16bit/np.abs(A))
    print rel_error16bit, rel_error
    print np.mean(abs_error16bit), np.mean(abs_error)
    assert np.mean(abs_error) < 0.02, "Compression error"
    assert rel_error < 0.03, "Compression error" 
    
 
 
def test_row_sum():
    for i in range(100):
        dims = np.random.randint(2,763,(2,))
        #dims = [128,1500]
        A = np.float32(np.random.rand(dims[0],dims[1]))        
        B1 = gpu.array(A)
        B2 = gpu.zeros((A.shape[0],))
        
        #gpu.tick("rowsum")
        gpu.sum_row(B1, B2)
        #gpu.tick("rowsum")
        #print B2.tocpu()
        #print np.sum(A,axis=1)
        #errors = np.sort(np.sqrt(((np.sum(A,axis=1)-B2.tocpu())**2)).flatten())[::-1][0:10]
        #print errors
        #print A.shape
        t.assert_almost_equal(np.sum(A,axis=1),B2.tocpu(),3,"row sum")
    #gpu.tock("rowsum")

  
def test_row_max():
    for i in range(1):
        dims = np.random.randint(2,763,(2,))
        A = np.float32(np.random.randn(dims[0],dims[1]))        
        B1 = gpu.array(A)
        B2 = gpu.zeros((A.shape[0],))
        
        gpu.max_row(B1, B2)
        #print B2.tocpu()
        #print np.sum(A,axis=1)
        #errors = np.sort(np.sqrt(((np.sum(A,axis=1)-B2.tocpu())**2)).flatten())[::-1][0:10]
        #print errors
        #print A.shape
        t.assert_almost_equal(np.max(A,axis=1),B2.tocpu(),3,"row max")
    
def test_1bit_compression():
    for i in range(10):
        dims = np.random.randint(2,637,(2,))
        dims = [dims[0], dims[1] + (32- (dims[1] % 32))]#we need a multiple of 32 dims
        A = np.float32(np.random.randn(dims[0],dims[1]))  
        B = gpu.array(A)
        val_with_errors = gpu.zeros_like(B)
        errors = gpu.zeros_like(B)
        maskPos = gpu.zeros_like(B)
        maskNeg = gpu.zeros_like(B)    
        avgpos = gpu.zeros((A.shape[0],))
        avgneg = gpu.zeros((A.shape[0],))
        pos_count = gpu.zeros((A.shape[0],))
        neg_count = gpu.zeros((A.shape[0],))
        quant = gpu.empty_uint_like(B)
        C = gpu.empty_like(B)
        
        for j in range(3):
            A = np.float32(np.random.randn(dims[0],dims[1]))
            B = gpu.array(A)
            gpu.compress_1bit(B, val_with_errors, errors, avgpos, avgneg, quant, maskPos, maskNeg, pos_count, neg_count)      
            #print errors.tocpu().sum()
            #print avgpos.tocpu().sum()
            #print avgneg.tocpu().sum()
            gpu.decompress_1bit(quant, errors, avgpos, avgneg, C)
            print np.abs(errors.tocpu()).sum()/float(A.size)            
            assert np.abs(errors.tocpu()).sum()/float(A.size) < 1.0, "quantization error too large!"
    
           
def test_16bit_compression():   
    for i in range(100):
        dims = np.random.randint(2,637,(2,))                
        A = np.float32(np.random.randn(dims[0],dims[1]))  
        B = gpu.array(A)
        C1 = gpu.empty_like(B)
        C2 = gpu.empty_ushort_like(B)
        
        gpu.compress_16bit(B, C2)
        gpu.decompress_16bit(C2, C1)
        t.assert_array_almost_equal(C1.tocpu(),A,2,"half-float compression")
        
def test_ticktock():
    A = rdm.rand(128,1024)
    w = rdm.rand(1024,512)
    out = gpu.zeros((128,512))
    out_split = gpu.zeros((128,512),2)
    sync_out = gpu.zeros((128,512),2)
    
    gpu.tick("tick-tock test")
    for i in range(300):
        gpu.dot(A,w, out)   
    ms100 = gpu.tock("tick-tock test")
    t.assert_(ms100 > 0.0, "tick-tock is not working")
    gpu.tick("tick-tock test")
    for i in range(3000):
        gpu.dot(A,w, out)   
    ms1000 = gpu.tock("tick-tock test")
    print ms1000
    assert (ms100*10)*0.9 < ms1000 and (ms100*10)*1.1 > ms1000, "tick-tock has no linear scaling"



def test_to_pinned(): 
    for i in range(100):
        dims = np.random.randint(2,637,(2,))  
        A1 = np.random.rand(dims[0],dims[1])
        A2 = np.copy(A1)
        pt_B = gpu.to_pinned_pointer(A1)
        B = gpu.pointer2ndarray(pt_B,(dims[0],dims[1]))
        A2 = np.float32(A2)
        t.assert_array_equal(A2, B, "pinned memory copy not working")
        
def test_empty_pinned():
    for i in range(10):
        dims = np.random.randint(2,637,(2,))
        A = np.float32(np.random.rand(dims[0],dims[1]))
        gpu.tick('to col-major pinned')
        pt_B = gpu.to_col_major_pinned_pointer(A)
        gpu.tick('to col-major pinned')
        B = gpu.pointer2ndarray(pt_B,(dims[0],dims[1]))
        
        t.assert_array_equal(A.T.flatten(),B.flatten())
        
    gpu.tock('to col-major pinned')
    
    
    
    
def test_printmat():
    A = np.random.rand(4,4)
    B = gpu.array(A)    
    gpu.printmat(B)
    gpu.printrows(B,2,4)
    gpu.printfull(B,0,2,2,4)
    print B
    
    
    A = np.random.rand(50,21)
    B = gpu.array(A)
    print B

if __name__ == '__main__':    
    nose.run()
    