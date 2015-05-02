'''
Created on Apr 30, 2015

@author: tim
'''
from neural_net import Neural_net
import numpy as np


X = np.float32(np.load('/home/tim/data/mnist/train_X.npy'))
y = np.float32(np.load('/home/tim/data/mnist/train_y.npy'))

net = Neural_net()
net.net.set_config_value('input_dropout',0.0)
net.net.set_config_value('dropout',0.0)
net.net.set_config_value('parallelism','data')
net.net.set_config_value('compression','32bit')
net.net.set_config_value('learnng_rate',0.001)
net.fit(X,y,batch_size=128)