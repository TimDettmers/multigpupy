'''
Created on Apr 20, 2015

@author: tim
'''
'''
Created on Apr 11, 2015

@author: tim
'''
from batch_allocator import batch_allocator
import time
from layer import *
import tsne
from scipy.spatial.distance import cdist


class Autoencoder(object):
    def __init__(self, workdir=None, classes=None, learning_rate=0.003, hidden_size= [1024], code_size =128, dropout=0.5, input_dropout=0.2, epochs=100, unit=Logistic, network_name='autoencoder'):        
        self.net = Layer(workdir=workdir,network_name=network_name)
        for size in hidden_size:
            self.net.add(Layer(size, unit()))
        self.net.add(Layer(code_size,Code()))       
        for size in hidden_size:
            self.net.add(Layer(size, unit()))
        
        self.net.set_config_value('parallelism','None')
        #self.net.set_config_value('dropout', dropout)
        #self.net.set_config_value('input_dropout', input_dropout)
        self.net.set_config_value('learning_rate', learning_rate)
        self.net.set_config_value('error_evaluation','regression')
        self.epochs = epochs
        self.labels = None
        
    def fit(self, X, y=None, cv_size=1.0-0.8571429, test_size=0.0, batch_size = 128):        
        self.alloc = batch_allocator(X,y, cv_size,test_size,batch_size)        
        self.alloc.net = self.net        
        self.net.add(Layer(X.shape[1],Linear()))
        self.net.set_config_value('parallelism','None')
        self.net.root.config['compression'] = '32bit'
        self.net.root.config['error_evaluation'] = 'regression'        
        self.net.root.config['learning_rate'] =self.net.root.prev_layer.config['learning_rate']/10.
        self.net.root.config['learning_rate_decay'] =self.net.root.prev_layer.config['learning_rate_decay']
        self.net.root.config['momentum'] =self.net.root.prev_layer.config['momentum']
        self.net.root.config['dropout_decay'] =self.net.root.prev_layer.config['dropout_decay']
        root = self.net.root
        while root.next_layer: 
            if type(root.funcs) == Code: break
            root = root.next_layer
        root.config['dropout'] = 0.0
        root.funcs.dropout = 0.0
        for epoch in range(self.epochs):            
            t0 = time.time()
            for i in self.alloc.train():
                if self.net.config['parallelism'] == 'data' and self.alloc.batch.shape[2] != batch_size: continue
                self.net.forward(self.alloc.batch,self.alloc.batch)
                self.net.backward()
                if self.net.config['parallelism'] != 'data':
                    self.net.weight_update()
            
            self.net.log('EPOCH: {0}'.format(epoch+1))
            for i in self.alloc.train(0.1):   
                self.net.forward(self.alloc.batch,self.alloc.batch_y, False)        
                self.net.accumulate_error()        
            self.net.print_reset_error()
            
            if cv_size > 0.0:
                for i in self.alloc.cv():
                    self.net.forward(self.alloc.batch,self.alloc.batch_y, False)
                    self.net.accumulate_error()
                self.net.print_reset_error('CV')
                self.net.end_epoch()
            #print time.time()-t0
            
    def get_codes(self, X):
        X_gpu = gpu.array(X)
        self.net.forward(X_gpu,None, False)    
        root = self.net
        while root.next_layer:
            if type(root.funcs) == Code: break 
            root = root.next_layer
            
        if type(root.funcs) == Code: return root.out.tocpu()
        else: print 'No code layer found!'
        
    def classify(self, codes, y, test_codes, proba=False, classes=10):
        preds = []
        X = gpu.array(codes)
        buffer = gpu.empty_like(X)      
        row_buffer = gpu.empty((X.shape[0],1))  
        for i, code in enumerate(test_codes):
            if i % 1000 == 0: print i
            row = np.zeros((classes,))
            vec = gpu.array(code)
            gpu.sub(X,vec,buffer)
            gpu.square(buffer, buffer)
            gpu.sum_row(buffer, row_buffer)
            gpu.sqrt(row_buffer,row_buffer)
            distance = row_buffer.tocpu()
            nearest = np.argsort(distance, axis=None)[1:5]
            if proba:
                for i in range(classes):
                    idx = np.where(y[nearest]==i)[0]            
                    if idx.shape[0] > 0:
                        row[i] += np.sum(np.exp(-distance[idx]))
                preds.append(row.tolist())
            else:
                preds.append(y[nearest])
            
            del vec           
            #distance = cdist(codes, np.matrix(code),'cos')
            #nearest = np.argsort(distance, axis=None)[1:50]
            #counts = np.bincount(np.int32(y[nearest].flatten()))
            #preds.append(np.argmax(counts))
        
        return np.array(preds)
            
        
    def print_codes(self, X, y):
        codes = self.get_codes(X)
        tsne.print_codes(self.net.workdir, codes[0:500], y[0:500])
        
        
