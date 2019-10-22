#!/usr/bin/env python
# coding: utf-8

# In[3]:
#Remember that using sckit.modelSelection works musch faster that our usual medhod of sample 
#so in case of any parcticle problem use model selection method fro sckitlearn

import pandas as pd
import numpy as np


# In[4]:


#Reading of data file

data= pd.read_csv("iris.data", header=0, names=["sepal_width","sepal_length","petal_width","petal_length","class"])


# In[55]:


#Data preparation

train= data.sample(frac=0.8,random_state=100)
X_train= train.drop("class",axis=1)
Y_train= train.drop(X_train.columns.difference(['class']),axis=1)
test= data.drop(train.index,axis=0)
X_test= test.drop("class",axis=1)
Y_test=test.drop(X_test.columns.difference(['class']),axis=1)


# In[56]:


#using Knn

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, np.ravel(Y_train)) 
  
accuracy = knn.score(X_test, np.ravel(Y_test)) 
print ("Accuracy achieved = ",accuracy) 

y= np.ravel(Y_test)

knn_predictions = knn.predict(X_test)

#tabular representation of tested data

print ("\n====================\nResults\n====================\n") 
for i in range(len(Y_test)):
    if(y[i]==knn_predictions[i]):
        print("Correct Prediction for ",y[i])
    else:
        print("XXXXX Wrong XXXXX-----> correct: ",y[i],"   -----> predicted: ",knn_predictions[i])


# In[58]:


#using svm
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, np.ravel(Y_train)) 
svm_predictions = svm_model_linear.predict(X_test) 

accuracy = svm_model_linear.score(X_test, np.ravel(Y_test)) 
print ("Accuracy achieved = ",accuracy) 

y= np.ravel(Y_test)

knn_predictions = svm_model_linear.predict(X_test)

#tabular representation of tested data

print ("\n====================\nResults\n====================\n") 
for i in range(len(Y_test)):
    if(y[i]==svm_predictions[i]):
        print("Correct Prediction for ",y[i])
    else:
        print("XXXXX Wrong XXXXX-----> correct: ",y[i],"   -----> predicted: ",svm_predictions[i])


# In[ ]:




