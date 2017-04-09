import numpy as np
import csv
import random
reader = csv.DictReader(open('train.csv'), delimiter=',')
test = []
for line in reader:
    s = int(line['Survived'])
    clas =int(line['Pclass'])
    sex = line['Sex']
    if sex.lower() is 'male':
        sex = 1
    else:
        sex = 2
    age = line['Age']
    if not age:
        age = 0
    else:
        age = float(age)/3.0
    sib =int(line['SibSp'])
    par = int(line['Parch'])
    emb = line['Embarked']
    if emb:
        emb = int(ord(emb))/3.0
    else:
        emb = 0
    test.append(np.matrix([s,clas,sex,age,sib,par,emb]))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


weights = np.ones(7,)
weights = 0.01*weights
test = random.sample(test,len(test))
traindata = test[:int(0.7*len(test))]
testdata = test[int(0.7*len(test)):]
last = 100
l = len(traindata)**-1
while last>0.005 :
    print("aa")
    for test in traindata:
        y = test[:,0]
        x = test[:,1:]
        x = np.append([1],x)

        z = np.dot(np.matrix.transpose(weights),x)
        h = sigmoid(z)
        s = float((y-h)*l*5)


        weights+=x*s
        last = s
    print(last)

print("done")
cost = 0

for data in testdata:
    y = data[:,0]
    # print(y,data)
    x = test[:, 1:]
    x = np.append([1], x)

    z = np.dot(np.matrix.transpose(weights), x)
    h = sigmoid(z)
    if h >= 0.5 :
        h = 1
    else :
        h = 0
    print(y,h,cost)
    cost+=(y-h)
print(cost,len(testdata))