# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Import KMeans and use for loop to cluster the data.
4. Predict the cluster and plot data graphs.
5. Print the output and end the program.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: HARSSHITHA LAKSHMANAN
RegisterNumber: 212223230075

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers_EX8 (2).csv')
data
x = data[['Annual Income (k$)','Spending Score (1-100)']]
x
plt.figure(figsize=(4, 4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.show()
#number of clusters
k = 5

#initialize hmeans
kmeans=KMeans(n_clusters=k)

#Fit the data
kmeans.fit(x)
centroids=kmeans.cluster_centers_
#get the cluster labels for each data point
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors = ['r', 'g', 'b', 'c', 'm']  # define colors for each cluster
k = 5
for i in range(k):
    cluster_points = x[labels == i]
    plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'],
                color=colors[i], label=f'Cluster {i+1}')
    distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
    radius = np.max(distances)
    circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
    plt.gca().add_patch(circle)
    plt.scatter(centroids[i,0],centroids[i,1], marker='*', s=200, color='k', label='Centroids')
plt.title('Kmeans Clustering')
plt.xlabel('Annual income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show() 
*/
```

## Output:
![K Means Clustering for Customer Segmentation](sam.png)
![Screenshot 2024-04-16 213002](https://github.com/harshulaxman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145686689/d95ade30-b20b-49ae-8c00-ab4f54a55c48)
![Screenshot 2024-04-16 213011](https://github.com/harshulaxman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145686689/616f0b20-2beb-4dac-92aa-9a523ff30754)
![Screenshot 2024-04-16 213020](https://github.com/harshulaxman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145686689/8e6add25-ba85-440f-a6c8-4351791a0ba5)
![Screenshot 2024-04-16 213030](https://github.com/harshulaxman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145686689/ccbadcfc-fadc-4519-bd35-34ad8571343b)
![Screenshot 2024-04-16 213038](https://github.com/harshulaxman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145686689/e6d45f74-4fda-40cf-a045-fbbd91e13b8c)
![Screenshot 2024-04-16 213052](https://github.com/harshulaxman/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145686689/b260a652-9447-42b0-8500-1054e0ef0155)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
