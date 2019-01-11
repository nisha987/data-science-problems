
# coding: utf-8

# In[1]:


# we aim to predict sale of product based when the ad is clicked online


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


data=pd.read_csv('Social_Network_Ads.csv')


# In[6]:


data.head(10)


# In[13]:


X=data.iloc[:,[2,3]].values


# In[14]:


y=data.iloc[:,4].values


# In[15]:


print(X.shape)


# In[52]:


print(y.shape)


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[26]:


from sklearn.preprocessing import StandardScaler


# In[27]:


s=StandardScaler()


# In[28]:


X_train=s.fit_transform(X_train)


# In[29]:


X_test=s.transform(X_test)


# In[34]:


print(X_train[0:5])


# In[35]:


print(X_test[0:5])


# In[37]:


from sklearn.neighbors import KNeighborsClassifier


# In[38]:


classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)


# In[40]:


classifier.fit(X_train, y_train)


# In[41]:


y_pred =classifier.predict(X_test)


# In[42]:


print(y_pred)


# In[44]:


c=0
for i in range(0,len(y_pred)):
    if(y_pred[i]==y_test[i]):
        c=c+1
accuracy=c/len(y_pred)
print("Accuracy is")
print(accuracy)


# In[49]:


from matplotlib.colors import ListedColormap


# In[50]:


X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[51]:


X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

