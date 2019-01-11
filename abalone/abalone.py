
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[14]:


df=pd.read_csv("abalone.csv")


# In[15]:


df.head(5)


# In[16]:


df['rings'].unique()


# In[17]:


df.describe()


# In[22]:


plt.hist(df['rings'],bins=30)

