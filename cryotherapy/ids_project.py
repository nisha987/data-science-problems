
# coding: utf-8

# In[136]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[137]:


df=pd.read_csv("Cryotherapy.csv", encoding='utf-8')


# In[138]:


df.head()


# In[139]:


df.describe()


# In[140]:


df.info()


# In[141]:


df.isnull().values.any()


# In[142]:


df['sex'].value_counts()


# In[143]:


df['Number_of_Warts'].value_counts()


# In[144]:


df['Type'].unique()


# In[145]:


df['Type'].value_counts()


# In[146]:


df['area_std']=df['Area']/10


# In[147]:


df['area_std'].describe()


# In[148]:


df.boxplot(grid=True, figsize=(10,10))


# In[149]:


df.hist(column='age', grid=True)


# In[150]:


df.hist( grid=True, figsize=(10,10))


# In[151]:


df.head()


# In[152]:


df.groupby('Type').sum().plot()


# In[153]:


df['age'].hist(by=df['Type'])


# In[154]:


df.groupby('Type').sum()


# In[155]:


df['Area'].hist(by=df['Type'])


# In[156]:


df['Number_of_Warts'].hist(by=df['Type'])


# In[157]:


df['Time'].hist(by=df['Type'])


# In[158]:


corr=df.corr(method='pearson')


# In[159]:


corr.style.background_gradient()


# In[160]:


mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

