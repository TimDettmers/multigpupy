'''
Created on Apr 11, 2015

@author: tim
'''
from batch_allocator import batch_allocator
import time
from layer import *

class Neural_net(object):
    def __init__(self, workdir=None, classes=10, learning_rate=0.003, hidden_size= [1024,1024], dropout=0.5, input_dropout=0.2, epochs=100, unit=Logistic, network_name='neural_net'):        
        self.net = Layer(workdir=workdir,network_name=network_name)
        for size in hidden_size:
            self.net.add(Layer(size, unit()))
        self.net.add(Layer(classes,Softmax()))       
        
        self.net.set_config_value('parallelism','None')
        self.net.set_config_value('dropout', dropout)
        self.net.set_config_value('input_dropout', input_dropout)
        self.net.set_config_value('learning_rate', learning_rate)
        self.epochs = epochs
        
    def fit(self, X, y, cv_size=0.2, test_size=0.0, batch_size = 128):        
        self.alloc = batch_allocator(X,y, cv_size,test_size,batch_size)        
        self.alloc.net = self.net
        for epoch in range(self.epochs):
            for i in self.alloc.train():
                self.net.forward(self.alloc.batch,self.alloc.batch_y)        
                self.net.backward()
                self.net.weight_update()
            
            self.net.log('EPOCH: {0}'.format(epoch+1))
            for i in self.alloc.train(0.1):   
                self.net.forward(self.alloc.batch,self.alloc.batch_y, False)        
                self.net.accumulate_error()        
            self.net.print_reset_error()
            
            for i in self.alloc.cv():
                self.net.forward(self.alloc.batch,self.alloc.batch_y, False)
                self.net.accumulate_error()
            self.net.print_reset_error('CV')
            self.net.end_epoch()
            
    def predict_proba(self, X):
        X_gpu = gpu.array(X)
        self.net.forward(X_gpu,None, False)
        return self.net.root.out.tocpu()        
        

    