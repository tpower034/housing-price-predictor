#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thomas Power
Date: February 28th, 2023
Description: In this file I will import the data, clean, look at the class variable, and run KNN.

"""

import numpy as np
import numpy
import random
import math
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)


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


#Here we can see the correlation matrix
corr_matrix_0 = data.corr()
series = pd.Series(corr_matrix_0['SalePrice'])


#Take a look at indexes over .4 correlation
important_variables_with_sales = series.index[(series > .40)]
#Take a look at indexes over .4 correlation and less than 1 so SalePrice is not included
important_variables = series.index[(1 > series) & (series > .40)]

#This is the data with just the important variables and our class SalesPrice
data_filtered = data[important_variables_with_sales].copy()

#kmeans
print("\n----------KMeans----------")
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(data_filtered)
    distortions.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('The Knee Method showing optimal k')
plt.show()

#From this we can see that optimal is 3

#We will randomly select two columns
print("\nThe important variables are: ")
print(important_variables)
columns = {'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',
       'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars',
       'GarageArea'}
random.seed(21)
print("\nThe random columns are " + str(random.sample(columns, 2)))

#We will take a look at GarageArea and TotalBsmtSF
data = data_filtered[['GarageArea', 'TotalBsmtSF']].copy()

kmeanModel = KMeans(n_clusters=3, random_state = 123)
predict = kmeanModel.fit_predict(data)

filtered_label0 = data[predict == 0]
filtered_label1 = data[predict == 1]
filtered_label2 = data[predict == 2]

#plotting the results
plt.scatter(filtered_label0['GarageArea'] , filtered_label0['TotalBsmtSF'], color = 'red', label ='points in cluster 1')
plt.scatter(filtered_label1['GarageArea'] , filtered_label1['TotalBsmtSF'], color = 'blue', label ='points in cluster 2')
plt.scatter(filtered_label2['GarageArea'] , filtered_label2['TotalBsmtSF'], color = 'black', label ='points in cluster 3')

#adding centroid
n_clusters = 3
colmap = {0:'red', 1:'blue', 2:'black'}
x = data[['GarageArea', 'TotalBsmtSF']].values
kmeans_classifier = KMeans(n_clusters = n_clusters, random_state = 123)
y_means = kmeans_classifier.fit_predict (x)
data['cluster'] = y_means
centroids = kmeans_classifier.cluster_centers_

for i in range ( n_clusters ):
    plt . scatter ( centroids [i][0] , centroids [i][1], color = colmap [i],
                   marker ='x', s=300 , label ='centroid' + str (i +1))
plt.legend ( loc ='upper right', prop={'size': 6})
plt.title('KMeans')
plt.xlabel ('TotalBsmtSF')
plt.ylabel ('GarageArea')
plt.show ()

#seeing what class the each cluster belongs to
print("\n----------Cluster 1----------")
data['class'] = data_filtered['SalePrice']
cluster1 = data[data['cluster'] == 0].copy()
cluster1_values = cluster1["class"].value_counts()
print("\nFor cluster 1, the percent in class 0 is " + str((cluster1_values[0]/len(cluster1))*100) + "%")
print("For cluster 1, the percent in class 1 is " + str((cluster1_values[1]/len(cluster1))*100) + "%")
print("For cluster 1, the centroid x value is:")
print(centroids[0][0])
print("For cluster 1, the centroid y value is:")
print(centroids[0][1])
print("Cluster 1 has a majority class 0")

print("\n----------Cluster 2----------")
cluster2 = data[data['cluster'] == 1].copy()
cluster2_values = cluster2["class"].value_counts()
print("\nFor cluster 2, the percent in class 0 is " + str((cluster2_values[0]/len(cluster2))*100) + "%")
print("For cluster 2, the percent in class 1 is " + str((cluster2_values[1]/len(cluster2))*100) + "%")
print("For cluster 2, the centroid x value is:")
print(centroids[1][0])
print("For cluster 2, the centroid y value is:")
print(centroids[1][1])
print("Cluster 2 has a majority class 0")

print("\n----------Cluster 3----------")
cluster3 = data[data['cluster'] == 2].copy()
cluster3_values = cluster3["class"].value_counts()
print("\nFor cluster 3, the percent in class 0 is " + str((cluster3_values[0]/len(cluster3))*100) + "%")
print("For cluster 3, the percent in class 1 is " + str((cluster3_values[1]/len(cluster3))*100) + "%")
print("For cluster 3, the centroid x value is:")
print(centroids[2][0])
print("For cluster 3, the centroid y value is:")
print(centroids[2][1])
print("Cluster 3 has a majority class 1")

class1_centroid = (centroids[0][0], centroids[0][1])
class2_centroid = (centroids[1][0], centroids[1][1])
class3_centroid = (centroids[2][0], centroids[2][1])
#data points
data_points = list(zip(data['GarageArea'],data['TotalBsmtSF']))
predicted_class = []
#calculating distances
for i in range(0,len(data)):
    class1_distance = math.dist(class1_centroid, data_points[i])
    class2_distance = math.dist(class2_centroid, data_points[i])
    class3_distance = math.dist(class3_centroid, data_points[i])
    if min(class1_distance, class2_distance, class3_distance) == class1_distance:
        predicted_class.append(0) #adding predicted class
    if min(class1_distance, class2_distance, class3_distance) == class2_distance:
        predicted_class.append(0) #adding predicted class
    if min(class1_distance, class2_distance, class3_distance) == class3_distance:
        predicted_class.append(1) #adding predicted class
        
data['predicted_class'] = predicted_class
total_classified_correctly = np.sum(data['predicted_class'] == data['class'])
accuracy = total_classified_correctly/len(data)
print("\n----------Accuracy----------\n")
print("The overall accuracy of this new classifier when applied to the complete data set accuracy is " + str(accuracy*100) + "%")

#This is for table
TP = 0
FP = 0
TN = 0
FN = 0
total_accuracy = 0
predicted_class = np.array(data['predicted_class'])
data = np.array(data_filtered['SalePrice'])

for i in range(0,len(data)):
    if predicted_class[i] == 1 and  predicted_class[i] == data[i]:
        TP = TP+1 #counting  positives
    if predicted_class[i] == 1 and  data[i] == 0:
        FP = FP+1 #counting false positives
    if predicted_class[i] == 0 and  predicted_class[i] == data[i]:
        TN = TN+1 #counting true negatives
    if predicted_class[i] == 0 and  data[i] == 1:
        FN = FN+1 #counting false negatives   
    if predicted_class[i] ==  data[i]:
        total_accuracy = total_accuracy+1 #counting total accuracy  
        
TPR = TP/(TP+FN)
TNR = TN/(TN+FP)
accuracy = total_accuracy/len(data)


print("\n----------- Summary -----------\n")
print("True positives for KMeans is: " + str(TP))
print("False positives for KMeans is: " + str(FP))
print("True negatives for KMeans is: " + str(TN))
print("False negatives for KMeans is: " + str(FN))
print("True positive for KMeans is: " + str(TPR))
print("True negative for KMeans is: " + str(TNR))

#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(data, predicted_class)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title("KMeans")
plt.show()

        
       
        
        
        