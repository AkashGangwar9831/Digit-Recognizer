# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:51:14 2019

@author: Akash
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

#import numpy as np
import pandas as pd

dataset_1=pd.read_csv('F:/digit-recognizer/train.csv')

dataset_2=pd.read_csv('F:/digit-recognizer/test.csv')

X=dataset_1.iloc[:,1:]
y=dataset_1.iloc[:,0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#KNN
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

y_pred_KNN=classifier.predict(dataset_2)
index=[]
for i in range(1,28000+1):
    index.append(i)

d={'ImageId':index,'Label':y_pred_KNN}
df=pd.DataFrame(d,index=None)
df.to_csv('F:/digit-recognizer/KNN_res.csv',index=False)

#SVM
from sklearn.svm import SVC
svclassifier=SVC(kernel='poly')
svclassifier.fit(X_train,y_train)
y_pred=svclassifier.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

y_pred_SVM=svclassifier.predict(dataset_2)
index=[]
for i in range(1,28000+1):
    index.append(i)

d={'ImageId':index,'Label':y_pred_SVM}
df=pd.DataFrame(d,index=None)
df.to_csv('F:/digit-recognizer/SVM_res.csv',index=False)