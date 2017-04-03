import random
import numpy as np
import math
class Model(object):
    def __init__(self,dataset,propotion,rate):
        self.dataset = dataset
        print(len(self.dataset))
        self.traindata=[]
        self.testdata=[]
        self.propotion = propotion
        self.setdata()
        self.weights=np.matrix(np.random.randn(1,6))
        self.rate = rate
        for i in range(len(self.weights)):
            self.weights[i]= self.weights[i]*0.01
        print(self.weights)

    def setdata(self):
        self.dataset = random.sample(self.dataset, len(self.dataset))
        # print(self.dataset)
        length_data = int(len(self.dataset)*self.propotion)
        self.traindata = self.dataset[:length_data]
        self.testdata = self.dataset[length_data:]

    def sigmoid(self,z):
        return  1/(1 + np.exp(-z))

    def cost(self,y,h):
        t = y * np.log(h) + (1-y) * np.log(1 - h)
        return t

    def buildModel(self):
       lent = len(self.traindata)
       i = 0
       for data in self.traindata:
           y = data[:,0]
           x = data[:,1:]

           z = np.dot(self.weights,np.matrix.transpose(x))
           h = self.sigmoid(z)
           s = np.ones(6,)
           s = np.multiply(np.matrix.transpose(s),(h-y)*self.rate)
           s = np.multiply(x,s)
           w_up = np.subtract(self.weights,s)
           self.weights = w_up
           print(w_up)
           # s = np.multiply()
           i+=1
           if i > 8:
                break