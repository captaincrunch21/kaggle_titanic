import random
import numpy as np
import math
import pickle
class Model_log(object):
    def __init__(self,dataset,propotion,rate):
        self.dataset = dataset
        print(len(self.dataset))
        self.traindata=[]
        self.testdata=[]
        self.propotion = propotion
        self.setdata()
        self.weights=np.matrix(np.random.randn(1,6))
        self.rate = rate

        self.weights = self.weights*0.01
        print(self.weights)

    def setdata(self):
        self.dataset = random.sample(self.dataset, len(self.dataset))
        print(len(self.dataset))
        length_data = int(len(self.dataset)*self.propotion)
        self.traindata = self.dataset[:length_data]
        self.testdata = self.dataset[length_data:]

    def sigmoid(self,z):
        return  1/(1 + np.exp(-z))

    def cost(self,y,h):
        return(math.fabs(y-h))
    def checktest(self):
        costnow = 0
        i = 0
        g = 0
        l = 0
        for data in self.testdata:
            y = data[:, 0]
            x = data[:, 1:]
            z = np.dot(self.weights, np.matrix.transpose(x))
            h = self.sigmoid(z)
            if h >= 0.5 :
                h = 1
                g+=1
            else:
                h =0
                l+=1

            costnow+=self.cost(y,h)
            # print(costnow,i)
            i+=1
        print(costnow,len(self.testdata),g,l)
        self.picklize()
    def buildModel(self):
        while(True):
            lent = len(self.traindata)
            i = 0
            print(i)
            i+=1
            for data in self.traindata:
                y = data[:,0]
                x = data[:,1:]

                z = np.dot(self.weights,np.matrix.transpose(x))
                h = self.sigmoid(z)
                s = np.ones(6,)
                s = np.multiply(np.matrix.transpose(s),(h-y)*self.rate*(lent**-1))
                s = np.multiply(x,s)
                w_up = np.subtract(self.weights,s)

                self.weights = w_up
                # print(w_up)
            if math.fabs(s[:,0]) < 0.005:
                break
        print('completed')
        self.checktest()


    def picklize(self):
        pickle.dump(self,open('self.pkl','wb'))
