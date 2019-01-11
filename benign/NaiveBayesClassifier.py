
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[3]:


class NaiveBayesClassifier(object):
    def __init__(self):
        pass
    
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
        
        self.no_of_classes = np.max(self.y_train)+1
        
    def euclidianDistance(self,Xtest, Xtrain):
        return np.sqrt(np.sum(np.power(Xtest-Xtrain),2))
    
    def predict(self, X, radius=0.4):
        pred =[]
        
        members_of_class=[]
        for i in range(self.no_of_classes):
            counter=0
            for j in range(len(self.y_train)):
                if self.y_train[j]==i:
                    counter+=1
            members_of_class.append(counter)
            
        for t in range(len(X)):
            prob_of_classes =[]
            
            for i in range(self.no_of_classes):
                prior_prob =members_of_class[i]/len(self.y_train)
                inRadius_no =0
                inRadius_no_current_class=0
                
                for j in range(len(self.X_train)):
                    if self.euclideanDistance(X[t], self.X_train[j]) < radius:
                        inRadius_no+=1
                        if self.y_train[j]==i:
                            inRadius_no_current_class+=1
                
                margin_prob = inRadius_no/len(self.X_train)
                
                likelihood = inRadius_no_current_class/len(self.X_train)
                
                post_prob =(likelihood* prior_prob)/margin_prob
                prob_of_classes.append(post_prob)
                
            pred.append(np.argmax(prob_of_classses))
        return pred


# In[4]:


def accuracy(y_tes, y_pred):
    correct =0;
    for i in range(len(y_pred)):
        if(y_tes[i]==y_pred[i]):
            correct+=1
            
    return (correct/len(y_tes))*100


# In[6]:


def breastCancerTest():
    dataset =pd.read_csv('breastCancer.csv')
    dataset.replace('?', 0, inplace=True)
    dataset = dataset.applymap(np.int64)
    X=dataset.iloc[:,1:-1].values
    y=dataset.iloc[:,-1].values
    
    y_new =[]
    for i in range(len(y)):
        if y[i]==2:
            y_new.append(0)
        else:
            y_new.append(1)
    y_new = np.array(y_new)
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)
    
    NB=NaiveBayesClassifier()
    NB.fit(X_train,y_train)
    
    y_pred =NB.predict(X_test,radius=8)
    
    from sklearn.naive_bayes import GaussianNB
    NB_sk=GaussianNB()
    NB_sk.fit(X_train,y_train)
    
    sk_pred = NB_sk.predict(X_test)
    
    print("accuracy for my Naive bayes classifier:", accuracy(y_test,y_pred),"%")
    print("accuracy for sklearn naive bayes classifier:",accuracy(y_test,sk_pred),"%")
    


# In[12]:


breastCancerTest()

