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
        age = float(age)/3.0

    sib =int(line['SibSp'])

    par = int(line['Parch'])

    emb = line['Embarked']
    if emb:
        emb = int(ord(emb))/3.0
    else:
        emb = 0
    test.append(np.matrix([s,clas,sex,age,sib,par,emb]))
print(len(test))
model = Model.Model_log(dataset=test,propotion=0.8,rate=1)
model.buildModel()