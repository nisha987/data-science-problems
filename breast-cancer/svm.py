
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sklearn as scikit
import sklearn.preprocessing
from sklearn import svm
from sklearn import metrics
import sklearn.utils.multiclass as checking
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# In[ ]:


def plot_linear(X, temp):
    clf=svm.SVC(kernel='linear')
    clf.fit(X,temp)
    prediction=clf.predict(X)
    
    w=clf.coef_[0]
    a=-w[0]/w[1]
    xx=np.linspace(10,19)
    yy=a*xx - (clf.interrupt_[0])/w[1]
    h0 = plt.plot(xx,yy,'k-')
    
    

