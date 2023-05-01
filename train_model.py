import pandas as pd
import numpy as np
import time
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score



df = pd.read_csv("MachineLearningCVE/PortScan.csv")

df =  df.drop_duplicates(keep="first")

integer = []
f = []
for i in df.columns[:-1]:
    if df[i].dtype == "int64": integer.append(i)
    else : f.append(i)

df[integer] = df[integer].astype("int32")
df[f] = df[f].astype("float32")
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]


x = df.drop([' Label'],axis=1)
y = df[' Label']

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.30,random_state=0)

print("X_train,Y_train", X_train.shape, Y_train.shape)
rus = RandomUnderSampler(random_state=0)
X_train, Y_train = rus.fit_resample(X_train, Y_train)

print("X_train,Y_train", X_train.shape, Y_train.shape)



bayes = GaussianNB()



imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

start = time.time()
bayes.fit(X_train, Y_train)
print("Time taken to train model: ", time.time()-start," seconds")

Predict_X =  bayes.predict(X_train)
scores = cross_val_score(bayes, X_train, Y_train, cv=5)
accuracy = metrics.accuracy_score(Y_train,Predict_X)
confusion_matrix = metrics.confusion_matrix(Y_train, Predict_X)
classification = metrics.classification_report(Y_train, Predict_X)


print()
print('--------------------------- Train Results --------------------------------')
print()
print ("Cross Validation Mean Score:" "\n", scores.mean())
print()
print ("Model Accuracy:" "\n", accuracy)
print()
print("Confusion matrix:" "\n", confusion_matrix)
print()
print("Classification report:" "\n", classification) 
print()



Predict_X =  bayes.predict(X_test)
scores = cross_val_score(bayes, X_test, Y_test, cv=7)
accuracy = metrics.accuracy_score(Y_test,Predict_X)
confusion_matrix = metrics.confusion_matrix(Y_test, Predict_X)
classification = metrics.classification_report(Y_test, Predict_X)

print()
print('--------------------------- Test Results --------------------------------')
print()
print ("Cross Validation Mean Score:" "\n", scores.mean())
print()
print ("Model Accuracy:" "\n", accuracy)
print()
print("Confusion matrix:" "\n", confusion_matrix)
print()
print("Classification report:" "\n", classification) 
print()



filename = 'model/bayes_model.sav'
pickle.dump(bayes, open(filename, 'wb'))