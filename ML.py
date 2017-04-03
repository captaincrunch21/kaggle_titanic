import csv
import Model
import numpy as np
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
        age = float(age)

    sib =int(line['SibSp'])

    par = int(line['Parch'])

    emb = line['Embarked']
    if emb:
        emb = int(ord(emb))
    else:
        emb = 0
    test.append(np.matrix([s,clas,sex,age,sib,par,emb]))
print(len(test))
model = Model.Model(dataset=test,propotion=0.9,rate=20)
model.buildModel()