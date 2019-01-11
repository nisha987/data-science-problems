
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("train.csv")


# In[3]:


df.head(10)


# In[4]:


df.describe()


# In[5]:


df['Property_Area'].value_counts()


# In[6]:


df['ApplicantIncome'].hist(bins=50)


# In[7]:


df.boxplot(column='ApplicantIncome')


# In[8]:


df.boxplot(column='ApplicantIncome', by='Education')


# In[9]:


df['LoanAmount'].hist(bins=50)


# In[10]:


df.boxplot(column='LoanAmount')


# In[11]:


df.apply(lambda x:sum(x.isnull()), axis=0)


# In[12]:


df['Self_Employed'].fillna('No',inplace=True)


# In[14]:


table=pd.pivot_table(df,values=["LoanAmount"], index=["Self_Employed"],columns=["Education"],aggfunc=[np.mean])


# In[15]:


table


# In[16]:


df['LoanAmount_log']=np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# In[17]:


df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)


# In[18]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


# In[19]:


from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le=LabelEncoder()
for i in var_mod:
    df[i]=le.fit_transform(df[i])
df.dtypes


# In[20]:


var_mod


# In[21]:


df.head(5)


# In[22]:


temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = pd.pivot_table(df,values='Loan_Status',index=['Credit_History'],aggfunc=lambda x:x.map({1:1,0:0}).mean())
print ('Frequency Table for Credit History:') 
print (temp1)

print ('\nProbility of getting loan for each Credit History class:')
print (temp2)


# In[23]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")


# In[24]:


temp3=pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[25]:


df.apply(lambda x: sum(x.isnull()),axis=0)


# In[26]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)


# In[27]:


df.apply(lambda x: sum(x.isnull()),axis=0)


# In[28]:


df['LoanAmount_log']=np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# In[29]:


df.head()


# In[30]:


df.apply(lambda x: sum(x.isnull()),axis=0)


# In[31]:


from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le=LabelEncoder()
for i in var_mod:
    df[i]=le.fit_transform(df[i])
df.dtypes


# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# In[48]:


def classification_model(model, data, predictors, outcome):
    model.fit(data[predictors],data[outcome])
    predictions=model.predict(data[predictors])
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    
    kf=KFold(data.shape[0],n_splits=5)
    error=[]
    for train, test in kf:
        train_predictors=(data[predictors].iloc[train,:])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
        error.append(model.score(data[predictors].iloc[test,:],data[outcome].iloc[test]))
        
    print("Cross validation score : %s" % "{0:.3%}".format(np.mean(error)))
    
    model.fit(data[predictors], data[outcome])


# In[49]:


outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, df,predictor_var,outcome_var)


# In[50]:


predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, df, predictor_var, outcome_var)


# In[53]:


model=DecisionTreeClassifier()
predictor_var=['Credit_History','Gender','Married','Education']
classification_model(model,df,predictor_var,outcome_var)


# In[54]:


predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model, df,predictor_var,outcome_var)


# In[57]:


model=RandomForestClassifier(n_estimators=100)
predictor_var=['Gender','Married','Dependents','Education','Self_Employed','Loan_Amount_Term','Credit_History','Property_Area','LoanAmount_log','TotalIncome_log']
classification_model(model,df,predictor_var,outcome_var)


# In[59]:


find_imp=pd.Series(model.features_importances_,index=predictor_var).sort_values(ascending=False)
print(find_imp)


# In[60]:


model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, df,predictor_var,outcome_var)

