'''
Created on Apr 30, 2015

@author: tim
'''
from neural_net import Neural_net
import numpy as np
import gpupy as gpu
import random_gpupy as rdm

X = np.float32(np.load('/home/tim/data/mnist/train_X.npy'))
y = np.float32(np.load('/home/tim/data/mnist/train_y.npy'))

'''
net = Neural_net('/home/tim/test_net')
net.net.set_config_value('input_dropout',0.0)
net.net.set_config_value('dropout',0.0)
net.net.set_config_value('parallelism','none')
net.net.set_config_value('compression','32bit')
net.net.set_config_value('learning_rate',0.001)
net.net.set_config_value('learning_rate_decay',0.98)
net.fit(X,y,batch_size=128)
'''


A = rdm.rand(128,100)
B = gpu.empty((128,100))

gpu.tick()
for i in range(10000):
    gpu.softmax(A)
gpu.tock()