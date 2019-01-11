
# coding: utf-8

#  # PREDICTING DIABETES : PRELIMINARY ANALYSIS AND CLASSIFICATION

# ### INTRODUCTION

# ###### 1. META INFORMATION

# The dataset is originally collected from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes or not, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# 
# 

# Released Under - https://www.niddk.nih.gov/

# Published On Data Portal - October 7, 2016
Source -https://www.kaggle.com/uciml/pima-indians-diabetes-database/home
# ###### 2. DATA QUALITY

# 1) Noise - No such extraneous object is shown that would modify with the original value.\n
# 2) Outliers - Outliers are observed in the box plot as explained later in the documentation.
# 3) Missing Values - There are no missing values.
# 4) Duplicate Data - No duplicate rows are present in the dataset.

# In[2]:


# we are importing libraries here

import pandas as pd
#pandas 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:




df=pd.read_csv("diabetes.csv")


# In[4]:


df.shape


# In[5]:


df.head(10)


# In[6]:


df.tail(5)


# In[10]:


df.info()


# In[11]:


df.isnull().sum()


# In[12]:


df.describe()


# In[13]:


def plot_corr(df,size=10):

    corr = df.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);


# In[16]:


fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
df.set_index('Glucose')
df['BloodPressure'].hist(bins=20)
plt.suptitle('BloodPressure vs Glucose', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.04, 'BloodPressure', ha='center')
fig.text(0.04, 0.5, 'Glucose', va='center', rotation='vertical')


# In[57]:


fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
df['Insulin'].hist()
plt.suptitle('Insulin vs Glucose', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.04, 'Insulin', ha='center')
fig.text(0.04, 0.5, 'Glucose', va='center', rotation='vertical')


# In[64]:


fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
df['SkinThickness'].hist()
plt.suptitle('SkinThickness vs Glucose', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.04, 'SkinThickness', ha='center')
fig.text(0.04, 0.5, 'Glucose', va='center', rotation='vertical')


# In[59]:


fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
df['BMI'].hist()
plt.suptitle('BMI vs Glucose', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.04, 'BMI', ha='center')
fig.text(0.04, 0.5, 'Glucose', va='center', rotation='vertical')


# In[60]:


fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
df['Glucose'].hist()
plt.suptitle('Glucose vs Pregnancy', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.04, 'Glucose', ha='center')
fig.text(0.04, 0.5, 'Pregnancy', va='center', rotation='vertical')


# In[61]:


fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
df['DiabetesPedigreeFunction'].hist()
plt.suptitle('DiabetesPedigreeFunction vs Glucose', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.04, 'DiabetesPedigreeFunction', ha='center')
fig.text(0.04, 0.5, 'Glucose', va='center', rotation='vertical')


# In[17]:


fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
df['Age'].hist()
plt.suptitle('Age vs Glucose', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.04, 'Age', ha='center')
fig.text(0.04, 0.5, 'Glucose', va='center', rotation='vertical')


# In[62]:


df.boxplot(grid=True, figsize=(10,10))


# In[81]:


plot_corr(df)


# In[79]:


df.corr()


# In[80]:


#del df['skin']


# In[65]:


df.shape


# In[66]:


plot_corr(df)


# In[71]:


#diabetes_map={True:1,False:0}
#df['Outcome']=df['Outcome'].map(diabetes_map)


# In[73]:


df.head(5)


# In[84]:


num_true= len(df.loc[df['Outcome']== 1])
num_false=len(df.loc[df['Outcome']== 0])
print("Number of True cases: {0} ({1:2.2f}%)".format(num_true,(num_true/(num_true+num_false))*100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false,(num_false/(num_true+num_false))*100))


# In[87]:


from sklearn.cross_validation import train_test_split
feature_col_names=['Pregnancies','Glucose','BloodPressure','SkinThickness','BMI','DiabetesPedigreeFunction','Age']
predicted_class_names=['Outcome']
X=df[feature_col_names].values # predictor feature columns (8xm)
y=df[predicted_class_names].values #predicted class (1=True,0=False) column (1xm)
split_test_size=0.30
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=split_test_size,random_state=42)
                # test_size =0.3 is 30%, 42 is the answer to everything


# ### We check to ensure we have the desired 70% train, 30% test split of the data

# In[88]:


print("{0:0.2f}% in training set".format((len(X_train)/len(df.index))*100))
print("{0:0.2f}% in test set".format((len(X_test)/len(df.index))*100))


# ### Verifying predicted value was splitted correctly

# In[90]:


print("Original True : {0} ({1:0.2f}%)".format(len(df.loc[df['Outcome']==1]),len(df.loc[df['Outcome']==1])/len(df.index)*100))
print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['Outcome']==0]),len(df.loc[df['Outcome']==0])/len(df.index)*100))
print("")
print("Training True : {0} ({1:0.2f}%)".format(len(y_train[y_train[:]==1]),len(y_train[y_train[:]==1])/len(y_train)*100))
print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:]==0]),len(y_train[y_train[:]==0])/len(y_train)*100))
print("")
print("Test True : {0} ({1:0.2f}%)".format(len(y_test[y_test[:]==1]),len(y_test[y_test[:]==1])/len(y_test)*100))
print("Test False : {0} ({1:0.2f}%)".format(len(y_test[y_test[:]==0]),len(y_test[y_test[:]==0])/len(y_test)*100))


# ### Post-split Data Preparation

# #### Hidden Missing Values

# In[91]:


df.head(5)


# #### How many rows have unexpected 0 values??

# In[93]:


print("# rows in dataframe: {0}".format(len(df)))
print("# rows missing glucose : {0}".format(len(df.loc[df['Glucose']==0])))
print("# rows missing diastolic_bp : {0}".format(len(df.loc[df['BloodPressure']==0])))
print("# rows missing thickness : {0}".format(len(df.loc[df['SkinThickness']==0])))
print("# rows missing insulin : {0}".format(len(df.loc[df['Insulin']==0])))
print("# rows missing bmi : {0}".format(len(df.loc[df['BMI']==0])))
print("# rows missing diab_pred : {0}".format(len(df.loc[df['DiabetesPedigreeFunction']==0])))
print("# rows missing age : {0}".format(len(df.loc[df['Age']==0])))


# ### Impute with mean

# In[94]:


from sklearn.preprocessing import Imputer
#Impute with mean all 0 values
fill_0 = Imputer(missing_values=0, strategy="mean",axis=0)
X_train= fill_0.fit_transform(X_train)
X_test= fill_0.fit_transform(X_test)


# ### Training with Initial Algorithm- Naive Bayes

# In[95]:


from sklearn.naive_bayes import GaussianNB
Nb_model=GaussianNB()
Nb_model.fit(X_train,y_train.ravel())
 


# ### Performance on training data

# In[96]:


Nb_predict_train= Nb_model.predict(X_train)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,Nb_predict_train))


# ### Performance on testing data

# In[97]:


Nb_predict_test=Nb_model.predict(X_test)
print(accuracy_score(y_test,Nb_predict_test))


# In[107]:


corr2=df.corr(method='spearman')


# In[106]:


mask = np.zeros_like(corr2, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr2, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

