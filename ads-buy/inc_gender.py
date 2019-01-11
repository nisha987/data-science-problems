
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data=pd.read_csv('Social_Network_Ads.csv')


# In[3]:


data.head(10)


# In[29]:


data.describe()


# In[4]:


data.apply(lambda x:sum(x.isnull()), axis=0)


# In[8]:


gen={"Female":1,"Male":0}
data['Gender']=data['Gender'].map(gen)


# In[9]:


data.head()


# In[25]:


p=data.corr(method='pearson')


# In[26]:


from string import ascii_letters
import seaborn as sns


# In[27]:


mask = np.zeros_like(p, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(p, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[28]:


data.boxplot(column='EstimatedSalary')


# In[30]:


data.boxplot(column='Age')


# In[40]:


plt.hist(data['EstimatedSalary'],bins=25, alpha=1)


# In[48]:


plt.hist(data['Age'], bins=10)


# In[49]:


data.head()


# In[51]:


X=data.iloc[:,[1,2,3]].values


# In[52]:


y=data.iloc[:,[4]].values


# In[54]:


print(X.shape)


# In[55]:


print(y.shape)


# In[56]:


from sklearn.model_selection import train_test_split


# In[135]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[136]:


from sklearn.preprocessing import StandardScaler


# In[137]:


s=StandardScaler()


# In[138]:


X_train=s.fit_transform(X_train)


# In[139]:


X_test=s.transform(X_test)


# In[140]:


from sklearn.neighbors import KNeighborsClassifier


# In[141]:


classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 3)


# In[142]:


classifier.fit(X_train, y_train)


# In[143]:


y_pred =classifier.predict(X_test)


# In[144]:


c=0
for i in range(0,len(y_pred)):
    if(y_pred[i]==y_test[i]):
        c=c+1
accuracy=c/len(y_pred)
print("Accuracy is")
print(accuracy)


# In[148]:


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

