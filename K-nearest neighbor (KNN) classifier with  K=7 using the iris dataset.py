#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # Using petal length and petal width
y = iris.target


# In[7]:


# Create and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X, y)


# In[8]:


# Create mesh to plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = .02  # Step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


# In[9]:


# Predict the decision boundaries
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('7-Nearest Neighbors Decision Boundaries with Iris Dataset')
plt.show()


# In[ ]:




