
'''
Created on Mar 21, 2015

@author: tim
'''
import nose
import gpupy as gpu
import random_gpupy as rdm
import numpy as np
import numpy.testing as t

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
    C = gpu.logisticGrad(B).tocpu()  
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
    
def test_Slice():
    '''
    S = gpu.emptySlice()
    assert S.batch.start == 0
    assert S.batch.stop == np.iinfo(np.int32).max
    assert S.map.start == 0
    assert S.map.stop == np.iinfo(np.int32).max
    assert S.row.start == 0
    assert S.row.stop == np.iinfo(np.int32).max
    assert S.col.start == 0
    assert S.col.stop == np.iinfo(np.int32).max
    
    S.setSliceValues([slice(1,3),slice(3,6),slice(10,20)])
    assert S.batch.start == 0
    assert S.batch.stop == np.iinfo(np.int32).max
    assert S.map.start == 1
    assert S.map.stop == 3
    assert S.row.start == 3
    assert S.row.stop == 6
    assert S.col.start == 10
    assert S.col.stop == 20
    
    S.setSliceValues([slice(1,5)])
    assert S.col.start == 1
    assert S.col.stop == 5
    
    
    S.setSliceValues([slice(None,None, None)])
    assert S.col.start == 0
    assert S.col.stop == np.iinfo(np.int32).max
    '''
    
    
    
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
    C = gpu.fsynchronizingAdd(B)   
    
    t.assert_array_almost_equal(C.tocpu(), A*2, 7, "Synchronizing add does not work!")
    C*=0
    gpu.fsynchronizingAdd(B,C)
    t.assert_array_almost_equal(C.tocpu(), A*2, 7, "Synchronizing add does not work!")
    
       
    
    
if __name__ == '__main__':    
    nose.run()