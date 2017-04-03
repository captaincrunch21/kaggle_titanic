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
        self.weights=np.matrix(np.random.randn(6,1))
        self.rate = rate
        for i in range(len(self.weights)):
            self.weights[i]= self.weights[i]*0.001
        print(self.weights)

    def setdata(self):
        self.dataset = random.sample(self.dataset, len(self.dataset))
        # print(self.dataset)
        length_data = int(len(self.dataset)*self.propotion)
        self.traindata = self.dataset[:length_data]
        self.testdata = self.dataset[length_data:]

    def sigmoid(self,z):
        try :
            return  1/(1 + math.exp(-z))
        except:
            return 1

    def cost(self,y,h):
        try:
            t = y * math.log10(h) + (1-y) * math.log10(1 - h)
            return t
        except:
            return 10

    def buildModel(self):
       lent = len(self.traindata)
       while(True):
            cost_now = 0
            print(cost_now)
            for data in self.traindata:

                y = data[:,0]
                th = data[:,1:]
                pro = np.dot(th,self.weights)
                h = self.sigmoid(pro)
                sub = np.multiply(np.ones(6,),self.rate*(h-y)*lent**-1)
                sub = np.multiply(sub,th)
                # print(sub)
                self.weights = np.subtract(self.weights,sub)
                # print(self.weights)
            for data in self.dataset:
                y= data[:,0]
                th = data[:,1:]
                h= self.sigmoid(np.dot(th,self.weights))
                # print((y,h))
                cost_now+=self.cost(y,h)

            if math.fabs(cost_now) < 0.5:
                print(cost_now)
                print('done')
                break
            else:
                print(cost_now)
                print('continue')