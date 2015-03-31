
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
    C1 = gpu.array(A)
    C2 = gpu.array(B)
    C = gpu.add(C1,C2)  
    out = gpu.empty(C.shape)      
    gpu.add(C1, C2, out)
    t.assert_array_almost_equal(C.tocpu(), A+B, 7, "Add not equal to numpy add!")
    t.assert_array_almost_equal(out.tocpu(), A+B, 7, "Add not equal to numpy add!")
    t.assert_array_almost_equal((C1+C2).tocpu(), A+B, 7, "Add not equal to numpy add!")
    C1+=C2
    t.assert_array_almost_equal(C1.tocpu(), A+B, 7, "Add not equal to numpy add!")
    
    A = np.random.rand(7,1,4)
    B = np.random.rand(7,1,4)
    C1 = gpu.array(A)
    C2 = gpu.array(B)
    C = gpu.add(C1,C2)        
    
    t.assert_array_almost_equal(C.tocpu(), A+B, 7, "Add not equal to numpy add!")
    
    
    
    
    
    
    
if __name__ == '__main__':    
    nose.run()