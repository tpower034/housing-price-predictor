#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thomas Power
Date: February 28th, 2023
Description: In this file I will import the data, clean, look at the class variable, and run naive bayes.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy
from scipy import stats
import seaborn as sns
from sklearn . naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from warnings import simplefilter
from sklearn import metrics
simplefilter(action='ignore', category=FutureWarning)


#Loading Data
data = pd.read_csv("train.csv")

#Cleaning data
#Removing columns that have more than 15% NAs
print("\n----------Eliminating Columns----------\n")
for columns in data:
    if (((data[columns].isna().sum())/len(data.axes[0]))*100) > 15:
        data = data.drop(columns, axis=1)
        print("Column " + columns + " has been removed from the dataframe due to over 15% of NA's present.")
data = data.dropna() #eliminating na rows
#Take a look at the histogram of SalesPrice
data['SalePrice'].hist()
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.title('Histogram of SalePrice')
plt.show()

#Next we will take a look at mean, median, and mode
print("\n----------Info on SalesPrice----------\n")
mean = numpy.mean(data['SalePrice'])
median = numpy.median(data['SalePrice'])
mode = stats.mode(data['SalePrice'])

print("\nThe mean of SalesPrice is $" + str(mean))
print("The median of SalesPrice is $" + str(median))
print("The mode of SalesPrice is $" + str(mode[0]))

data['SalePrice'] = np.where(data["SalePrice"] > 186761, 1, 0) #converting SalePrice to 1/0 based on mean


#Here we are forming correlation matrix
corr_matrix_0 = data.corr()
series = pd.Series(corr_matrix_0['SalePrice'])

#Next we will look at at a heatmap
sns.heatmap(corr_matrix_0)
plt.title('Heatmap of Correlation')
plt.show

#Here we can see sorted correlation
print("\n----------Sorted Correlation----------\n")
print(series.sort_values())

#Take a look at indexes over .4 correlation 
important_variables_with_sales = series.index[(series > .40)]
#Take a look at indexes over .4 correlation and less than 1 so SalePrice is not included
important_variables = series.index[(1 > series) & (series > .40)]
print("\n----------Important Variables To Focus On----------\n")
print(important_variables)

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

#Naive Bayes
NB_classifier = GaussianNB().fit (X_train ,Y_train)

#Predicting
prediction = NB_classifier.predict(X_train)

#Calculating accuracy
print("\n----------Accuracy----------\n")
total_classified_correctly = np.sum(prediction == Y_test)
accuracy = total_classified_correctly/len(Y_test)
print("The accuracy of Naive Bayes is " + str(accuracy*100) + "%")

#This is for table
print("\n----------Naive Bayes----------")
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
print("True positives for naive bayes is: " + str(TP))
print("False positives for naive bayes is: " + str(FP))
print("True negatives for naive bayes is: " + str(TN))
print("False negatives for naive bayes is: " + str(FN))
print("True positive for naive bayes is: " + str(TPR))
print("True negative for naive bayes is: " + str(TNR))

#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(Y_test, prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title("Naive Bayes")
plt.show()
