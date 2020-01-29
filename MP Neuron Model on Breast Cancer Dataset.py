import sklearn.datasets #importing datasets from sklearn 
import numpy as np
import pandas as pd

#loading breast cancer dataset to check if there is cancer or not, malignant or benign
breast_cancer = sklearn.datasets.load_breast_cancer() 
#loading into x the features on which one can check the cancer
x = breast_cancer.data 
#loading the target/output value, or to say a certain label
y = breast_cancer.target 

#print(x)
#print(y)
#print(x.shape, y.shape)

#Every dataset in skealrn has this feature_names which is a list of the column names.
data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names) 
#Adding a new column as output which shows the output as 0( if benign) or 1 (if malignant)
data['class'] = breast_cancer.target 

#print(data.head())
#print(data.describe())
#print(data['class'].value_counts()) #Show no. of values of class grouped into common terms
#print(breast_cancer.target_names) #Shows corresponding names

#mean of each feature grouped by class acrosst he different rows
#print(data.groupby('class').mean())



from sklearn.model_selection import train_test_split

X = data.drop('class', axis = 1)
Y = data['class']

#set size of the training data as 10% , if this parameter is not given it takes 0.25
#stratify makesthe mean data same so as to split the data correctly to make minimal differences
#random_state gives the seed, or to say a constant number, so that the test data doesn't change always and remains constant since we want our model to be reproducable 
X_train, X_test, Y_train, Y_test = train_test_split(X , Y, test_size = 0.3, stratify = Y, random_state = 1)


import matplotlib.pyplot as plt
plt.plot(X_train.T, '*')
#to make the x axis tags vertical, since they overwrite each other, we set rotation
plt.xticks(rotation = 'vertical')
plt.show()

#binarisation of data - converting data to binary format, 0 and 1 only, for any particular class
X_binarised_3_train = X_train['mean area'].map(lambda x: 0 if x < 1000 else 1)
#plt.plot(X_binarised_3_train, '*')

#For doing binarisation across all the elements/columns, we use pandas
X_binarised_train = X_train.apply(pd.cut, bins = 2, labels = [1,0])
#plt.plot(X_binarised_train.T, '*')
#plt.xticks(rotation = 'vertical')
#plt.show()
X_binarised_test = X_test.apply(pd.cut, bins = 2, labels = [1,0])
#above labels were made 1,0 as for 0,1 we see mean values of 0 more than 1 in general, so there's a problem with binarization, hence we convert to 1,0, as we see that in general case of malignancy the mean radius/texture etc. should be higher, which is lower in this case.
#print(type(X_binarised_test))
X_binarised_test = X_binarised_test.values
X_binarised_train = X_binarised_train.values
#Convert the dataframe type to array type 
#print(type(X_binarised_test))

# MP neuron model- Inference and searching
from random import randint
"""
b = 3

i = randint(0, X_binarised_train.shape[0])

print("For row", i)
if(np.sum(X_binarised_train[i, :]) >= b):
    print("MP Neuron inference is malign")
else:
    print("MP Neuron Inference is benign")
if(Y_train[i] == 1):
    print("Ground truth is malignant")
else:
    print("Ground truth is benign")
""" 
#Accuracy check and finding the inference
"""
b = 15
 
Y_pred_train = []
accurate_rows = 0

for x,y in zip(X_binarised_train, Y_train):
    y_pred = (np.sum(x) >= b)
    Y_pred_train.append(y_pred)
    accurate_rows += (y == y_pred)

print(accurate_rows , accurate_rows/X_binarised_train.shape[0])

for b in range(X_binarised_train.shape[1] + 1):
    Y_pred_train = []
    accurate_rows = 0

    for x,y in zip(X_binarised_train, Y_train):
        y_pred = (np.sum(x) >= b)
        Y_pred_train.append(y_pred)
        accurate_rows += (y == y_pred)
    print(b , accurate_rows , accurate_rows/X_binarised_train.shape[0]) 
"""
#testing the trained model on training data, i.e., using the found out b=28 value on the new dataset
from sklearn.metrics import accuracy_score

b = 28

Y_pred_test = []

for x in X_binarised_test:
    y_pred = (np.sum(x) >= b)
    Y_pred_test.append(y_pred)
    
accuracy = accuracy_score(Y_pred_test, Y_test)
print(b, accuracy)