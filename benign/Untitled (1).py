
# coding: utf-8

# In[70]:




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[71]:


class NaiveBayesClassifier(object):
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        self.no_of_classes = np.max(self.y_train) + 1
     
  
    def euclidianDistance(self, Xtest, Xtrain):
        return np.sqrt(np.sum(np.power((Xtest - Xtrain), 2)))
    
   
    def predict(self, X, radius=0.4):   
        pred = []
        
 
        members_of_class = []
        for i in range(self.no_of_classes):
            counter = 0
            for j in range(len(self.y_train)):
                if self.y_train[j] == i:
                    counter += 1
            members_of_class.append(counter)
        
   
        for t in range(len(X)):
      
            prob_of_classes = []
          
            for i in range(self.no_of_classes):
                
                prior_prob = members_of_class[i]/len(self.y_train)
                
                inRadius_no = 0
                inRadius_no_current_class = 0
                
                for j in range(len(self.X_train)):
                    if self.euclidianDistance(X[t], self.X_train[j]) < radius:
                        inRadius_no += 1
                        if self.y_train[j] == i:
                            inRadius_no_current_class += 1
                
                margin_prob = inRadius_no/len(self.X_train)
                
                likelihood = inRadius_no_current_class/len(self.X_train)
                
                
                post_prob = (likelihood * prior_prob)/margin_prob
                prob_of_classes.append(post_prob)
            
            pred.append(np.argmax(prob_of_classes))
                
        return pred


# In[72]:


def accuracy(y_tes, y_pred):
    correct = 0
    for i in range(len(y_pred)):
        if(y_tes[i] == y_pred[i]):
            correct += 1
    return (correct/len(y_tes))*100


# In[78]:


def breastCancerTest():

    dataset = pd.read_csv('breastCancer.csv')
    dataset.replace('?', 0, inplace=True)
    dataset = dataset.applymap(np.int64)
    X = dataset.iloc[:, 1:-1].values    
    y = dataset.iloc[:, -1].values
 
    y_new = []
    for i in range(len(y)):
        if y[i] == 2:
            y_new.append(0)
        else:
            y_new.append(1)
    y_new = np.array(y_new)

    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    NB = NaiveBayesClassifier()
    NB.fit(X_train, y_train)
    
    y_pred = NB.predict(X_test, radius=7.8)
    
    from sklearn.naive_bayes import GaussianNB
    NB_sk = GaussianNB()
    NB_sk.fit(X_train, y_train)
    
    sk_pred = NB_sk.predict(X_test)
    
    from sklearn.naive_bayes import BernoulliNB
    NB_sb = BernoulliNB()
    NB_sb.fit(X_train, y_train)
    
    sb_pred = NB_sb.predict(X_test)
     
    
    print("Accuracy for my Naive Bayes Classifier: ", accuracy(y_test, y_pred), "%")
    print("Accuracy for sklearn Naive Bayes Classifier: ",accuracy(y_test, sk_pred), "%")
    print("Accuracy for sklearn Naive Bayes Classifier bernoulli: ",accuracy(y_test, sb_pred), "%")
   


# In[79]:


breastCancerTest()

