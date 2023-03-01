#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thomas Power
Date: February 28th, 2023
Description: In this file I will import the data, clean, look at the class variable, and run Random Forest.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn . ensemble import RandomForestClassifier
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


#Loading Training Data
data = pd.read_csv("train.csv")


#cleaning data
#Removing columns that have more than 15% NAs
for columns in data:
    if (((data[columns].isna().sum())/len(data.axes[0]))*100) > 15:
        data = data.drop(columns, axis=1)
        
data = data.dropna() #eliminating na rows


#Next we will take a look at mean, median, and mode
mean = numpy.mean(data['SalePrice'])
data['SalePrice'] = np.where(data["SalePrice"] > 186761, 1, 0) #converting SalePrice to 1/0 based on mean


#Here we can see the correlation matrix
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

#create scaler here
scaler = StandardScaler()
scaler.fit(X_train) # fit the Scaler on x_train
X_train = scaler.transform(X_train) # apply the Scaler to x_train

X_test = scaler.transform(X_test) # apply the Scaler to the test data 


# Random Forest
print("\n----------Random Forest----------")
x_value_graph = []
error_rate = []
# running random forest on N 1-10 and max depth 1-5
for n_estimators in range(1,11):
    for max_depth in range(1,6):
        model = RandomForestClassifier ( n_estimators = n_estimators, 
                                max_depth = max_depth, criterion ='entropy', random_state=(123))
        model.fit (X_test, Y_test)
        #new instance
        test_instance = np.asmatrix(X_test)
        prediction = model.predict (test_instance)
        total_classified_correctly = np.sum(prediction == Y_test)
        accuracy = total_classified_correctly/len(Y_test)
        x_value_graph.append(max_depth)
        error_rate.append((1-accuracy)*100)
        print(f"The error rate of Random Forest with n value {n_estimators=} and depth of {max_depth=} is {str((1-accuracy)*100)}%")

#splitting up data to plot
plot_data = pd.DataFrame({'Max Depth':x_value_graph, 'Error Rate':error_rate})
n_1 = plot_data[0:5]
n_2 = plot_data[5:10]
n_3 = plot_data[10:15]
n_4 = plot_data[15:20]
n_5 = plot_data[20:25]
n_6 = plot_data[25:30]
n_7 = plot_data[30:35]
n_8 = plot_data[35:40]
n_9 = plot_data[40:45]
n_10 = plot_data[45:50]

#plotting

plt.plot('Max Depth', 'Error Rate', data=n_1, linestyle='-', marker='o', label="N1")
plt.plot('Max Depth', 'Error Rate', data=n_2, linestyle='-', marker='o', label="N2")
plt.plot('Max Depth', 'Error Rate', data=n_3, linestyle='-', marker='o', label="N3")
plt.plot('Max Depth', 'Error Rate', data=n_4, linestyle='-', marker='o', label="N4")
plt.plot('Max Depth', 'Error Rate', data=n_5, linestyle='-', marker='o', label="N5")
plt.plot('Max Depth', 'Error Rate', data=n_6, linestyle='-', marker='o', label="N6")
plt.plot('Max Depth', 'Error Rate', data=n_7, linestyle='-', marker='o', label="N7")
plt.plot('Max Depth', 'Error Rate', data=n_8, linestyle='-', marker='o', label="N8")
plt.plot('Max Depth', 'Error Rate', data=n_9, linestyle='-', marker='o', label="N9")
plt.plot('Max Depth', 'Error Rate', data=n_10, linestyle='-', marker='o', label="N10")
plt.title('Random Forest Error Rate')
plt.xlabel('Max Depth')
plt.ylabel('Error Rate')
plt.legend(loc=3, prop={'size': 8})

#The best results is n=10 and max depth of 5
model = RandomForestClassifier ( n_estimators = 7, 
                                max_depth = 5, criterion ='entropy', random_state=(123))
model.fit (X_test, Y_test)
test_instance = np.asmatrix(X_test)
prediction = model.predict (test_instance)
total_classified_correctly = np.sum(prediction == Y_test)
accuracy = total_classified_correctly/len(Y_test)

print("\nFrom this the best performance of Random Forest is with n value 7 and depth of 5")

print("\n----------Accuracy----------\n")
print(f"The accuracy of Random Forest with n value 7 and depth of 5 is {str((accuracy)*100)}%")

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
print("True positives for random forest is: " + str(TP))
print("False positives for random forest is: " + str(FP))
print("True negatives for random forest is: " + str(TN))
print("False negatives for random forest is: " + str(FN))
print("True positive for random forest is: " + str(TPR))
print("True negative for random forest is: " + str(TNR))

#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(Y_test, prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title("Random Forest")
plt.show()





