# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 07:57:12 2020

@author: Yashasvi Bajpai
"""

# Classification problem with ANN
# Customers exiting the Bank and predicting their exit possibility.

# This code is included with some quick helpful tips and notes. 


# Data Preprocessing


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset

dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values


# Encoding Categorical Data


# 1st and last column are categorical variables, and we need to encode them to numbers
# to work easily


# In this case, we'll encode country and gender name to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 

# #for countries
# ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
# x = ct.fit_transform(x)

# #for gender
# labelencoder_x = LabelEncoder()
# x[:, 2] = labelencoder_x.fit_transform(x[:, 2])


labelencoder_x_1 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2 = LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])
# onehotencoder = OneHotEncoder(categorical_features=[1])
# x = onehotencoder.fit_transform(x).toarray()


# dummy var for country
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)

# Avoiding dummy var trap 
x = x[:,1:]


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



# Splitting dataset into Test Set and Training Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Making the ANN
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN: Defining sequences of layers or defining a graph
# here it is a seqn. of layers

classifier = Sequential()

## Rectifier : Hidden layers | Signoid : Output


#### I later included dropout for tuning purposes ###
# Adding input and 1st hidden layer

# TIP: no. of nodes in hidden layer = avg(node in input layer and output layer)
# else use parameter tuning like cross-validation
# imput_dim is no of nodes in Input layer = no. of input variables
# But in subsequent hidden layers, its not needed as they know already.
classifier.add(Dense(units =6, kernel_initializer= 'uniform',activation='relu',input_dim = 11 ))
classifier.add(Dropout(rate= 0.1))
# rate should be least, but can be increased in case of overfitting.


# Adding the secong hidden layer
classifier.add(Dense(units =6, kernel_initializer= 'uniform',activation='relu'))
classifier.add(Dropout(rate= 0.1))
# Adding the Output Layer

classifier.add(Dense(units =1, kernel_initializer= 'uniform',activation='sigmoid'))
# Sigmoid best for probabilities,

# Compiling the ANN: Applying Stochastic GD on NN
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics= ['accuracy'] )
# Args:
    # optimizer : Algorithm
    # loss : Loss func ( like sum of squares )
    # metrics : Accuracy criterion to improve our ANN

# Fitting ANN to training set

classifier.fit(x_train, y_train,batch_size=10, epochs=100)
# The training line is above !






###################################################################################
# # Fitting Logistic regression to the training set

# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state=0)
# classifier.fit(x_train,y_train)
###################################################################################


# Predicting the Test Set Results

y_pred = classifier.predict(x_test)
y_pred = (y_pred> 0.5) # Setting a threshold value of the bank customers in concern for bank 

# Making Confusion Matrix : Contains our predictions right or wrong
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

###############
# Predicting a new single customer record
'''Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000 '''

###############


# Evaluating our ANN : Using k-fold Cross Validation
# Wrapping k-F CV into keras model, as it belongs to scikit

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense # Reimporting for modularity of code

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units =6, kernel_initializer= 'uniform',activation='relu',input_dim = 11 ))
    classifier.add(Dense(units =6, kernel_initializer= 'uniform',activation='relu'))
    classifier.add(Dense(units =1, kernel_initializer= 'uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics= ['accuracy'] )
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10 , epochs = 100)
# n_jobs to set no. of CPU for the task for training. -1 = All CPUs
# hence our computations run faster due to parallel computation
# cv  denotes no. of folds

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv= 10, n_jobs= -1)

mean = accuracies.mean()
variance = accuracies.std()

# Finding perfect value and very less variance above.

##############################################################################
# Improving the ANN
# here, we hope to reach 85% accuracy. Currently at 83.8875%

# Using Drop-out technique for improvement.
# Updated the layers with dropout fot improvaemnt // Check Code above

#-----------------------------------------------------------------------------#
# Tuning our Hyperparameters
# Using GridSearch technique

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units =6, kernel_initializer= 'uniform',activation='relu',input_dim = 11 ))
    classifier.add(Dense(units =6, kernel_initializer= 'uniform',activation='relu'))
    classifier.add(Dense(units =1, kernel_initializer= 'uniform',activation='sigmoid'))
    #classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics= ['accuracy'] )
    classifier.compile(optimizer = optimizer, loss= 'binary_crossentropy', metrics= ['accuracy'] )
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# Creating a Dict for hyperparameters to tune
parameters = {'batch_size':[25,32], 
              'epochs':[100,500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator= classifier,
                           param_grid=parameters,
                           scoring - 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(x_train,y_train)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_ 

# WARNING!!! Grid Search process will be very long and hence be cautious
# A few hours
# Estimated values : (25,500,rmsprop) ; 85.0624999999

###############################################################################
    
    
    