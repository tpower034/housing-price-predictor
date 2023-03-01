#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thomas Power
Date: February 28th, 2023
Description: In this file I will import the data, clean, look at the class variable, and run KNN.

"""

import numpy as np
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


#Loading Training Data
data = pd.read_csv("train.csv")


#Cleaning data
#Removing columns that have more than 15% NAs
for columns in data:
    if (((data[columns].isna().sum())/len(data.axes[0]))*100) > 15:
        data = data.drop(columns, axis=1)
        
data = data.dropna() #eliminating na rows


#Next we will take a look at mean, median, and mode
mean = numpy.mean(data['SalePrice'])
data['SalePrice'] = np.where(data["SalePrice"] > 186761, 1, 0) #converting SalePrice to 1/0 based on mean


#Here we are forming correlation matrix
corr_matrix_0 = data.corr()
series = pd.Series(corr_matrix_0['SalePrice'])


#Take a look at indexes over .4 correlation
important_variables_with_sales = series.index[(series > .40)]
#Take a look at indexes over .4 correlation and less than 1 so SalePrice is not included
important_variables = series.index[(1 > series) & (series > .40)]

#This is the data with just the important variables and our class SalesPrice
data_filtered = data[important_variables_with_sales].copy()


#Defining x and y
X = data_filtered[important_variables].values # defining X
Y = data_filtered['SalePrice'].values 

#Splitting training and testing
X_train ,X_test , Y_train , Y_test = train_test_split (X,Y,
                          test_size =0.5 , random_state = 123)

#Create scaler here
scaler = StandardScaler()
scaler.fit(X_train) # fit the Scaler on x_train
X_train = scaler.transform(X_train) # apply the Scaler to x_train


X_test = scaler.transform(X_test) # apply the Scaler to the test data 
X_test = pd.DataFrame(X_test)


#Knn for 1-50
print("\n----------KNN----------")
k_value = []
accuracy_rate = []
for k in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    prediction = knn.predict(X_test.values)
    total_classified_correctly = np.sum(prediction == Y_test)
    accuracy = total_classified_correctly/len(Y_test)
    k_value.append(k)
    accuracy_rate.append(accuracy*100)
    print(f"The accuracy of KNN with n value {k=} is {str((accuracy)*100)}%")

plot_data = pd.DataFrame({'K Value':k_value, 'Accuracy Rate':accuracy_rate})
plt.plot('K Value', 'Accuracy Rate', data=plot_data, linestyle='-', marker='o', label="K Value and Accuracy Rate")
plt.title('K Accuracy Chart')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.show()

print("\nFrom this the best performance of KNN is when k = 5")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test.values)
print("\n----------Accuracy----------\n")
total_classified_correctly = np.sum(prediction == Y_test)
accuracy = total_classified_correctly/len(Y_test)
print(f"The accuracy of KNN with n value k=5 is {str((accuracy)*100)}%")


#This is for table

TP = 0
FP = 0
TN = 0
FN = 0
total_accuracy = 0
for i in range(0,len(Y_test)):
    if prediction[i] == 1 and  prediction[i] == Y_test[i]:
        TP = TP+1 #counting  positives
    if prediction[i] == 1 and  Y_test[i] == 0:
        FP = FP+1 #counting false positives
    if prediction[i] == 0 and  prediction[i] == Y_test[i]:
        TN = TN+1 #counting true negatives
    if prediction[i] == 0 and  Y_test[i] == 1:
        FN = FN+1 #counting false negatives   
    if prediction[i] ==  Y_test[i]:
        total_accuracy = total_accuracy+1 #counting total accuracy  
TPR = TP/(TP+FN)
TNR = TN/(TN+FP)
accuracy = total_accuracy/len(Y_test)

print("\n----------- Summary -----------\n")
print("True positives for KNN with k = 5 is: " + str(TP))
print("False positives for KNN with k = 5 is: " + str(FP))
print("True negatives for KNN with k = 5 is: " + str(TN))
print("False negatives for KNN with k = 5 is: " + str(FN))
print("True positive for KNN with k = 5 is: " + str(TPR))
print("True negative for KNN with k = 5 is: " + str(TNR))

#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(Y_test, prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title("KNN")
plt.show()

