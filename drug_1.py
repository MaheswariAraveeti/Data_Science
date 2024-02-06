#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import PowerTransformer,FunctionTransformer
import streamlit as s


# In[2]:


data = pd.read_csv(r"C:\Users\Welcome\Machine Learning\Daily Tasks\drug200.csv")


# In[3]:


df = data.copy()


# In[4]:


df.info()


# In[5]:


df["BP"] = data["BP"].astype("category")
df["Cholesterol"] = data["Cholesterol"].astype("category")


# In[6]:


df.duplicated().sum()


# In[7]:


df.isnull().sum()


# In[8]:


df["BP"].unique()


# In[9]:


df["Cholesterol"].unique()


# In[10]:


df["Drug"].unique()


# In[11]:


fv = df.iloc[:,:-1]    # split data into features and class variables
cv = df.iloc[:,-1]


# In[12]:


#fv


# In[13]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()   
cv = lb.fit_transform(cv)


# In[14]:


#cv


# In[15]:


# find the relation between features
# Correlation matrix
#corr_matrix = fv.corr()
#corr_matrix


# In[16]:


#sns.heatmap(corr_matrix,annot=True) # find the relation between features
#plt.show()


# In[17]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,FunctionTransformer
from sklearn.impute import SimpleImputer


# In[18]:


# split the data into train and test based on fv and cv
x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,stratify = cv)


# In[19]:


numerical_data = x_train.select_dtypes(include=["int64","float64"])
categorical_data = x_train.select_dtypes(include=["object"])
ordinal_data =  x_train.select_dtypes(include=["category"])


# In[20]:


#ordinal_data


# In[21]:


num_p = Pipeline([("imp_n",SimpleImputer()),("scaling",StandardScaler())])


# In[22]:


cat_p = Pipeline([("imp_n",SimpleImputer(strategy='most_frequent')),("ohe",OneHotEncoder(sparse_output=False,drop ="first"))])


# In[23]:


ord_p = Pipeline([('Null_values_imputation_2', SimpleImputer(strategy='most_frequent')),
                         ('Ordinal_Encoding', OrdinalEncoder(categories= [['NORMAL','LOW','HIGH'],['NORMAL','HIGH']]))])


# In[24]:


ctp = ColumnTransformer([("num",num_p,numerical_data.columns),("cat",cat_p,categorical_data.columns),(["ord",ord_p,ordinal_data.columns])],remainder="passthrough")


# In[25]:


finalp =  Pipeline([("preprocess",ctp)]) 


# In[26]:


#finalp


# In[27]:


finalp.fit_transform(x_train)


# In[28]:


x_train_fit = ctp.fit_transform(x_train)
x_test_trans = ctp.transform(x_test)


# In[32]:


ctp.get_feature_names_out()


# In[30]:


from mixed_naive_bayes import MixedNB


# In[33]:


mb = MixedNB(categorical_features= [2,3,4])
model = mb.fit(x_train_fit, y_train)


# In[35]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[37]:


accuracy_score(y_test,model.predict(x_test_trans))


# In[46]:


import pickle
pickle.dump(finalp,open(r"C:\Users\Welcome\Machine Learning\Daily Tasks\final_drug1","wb"))
pickle.dump(model,open(r"C:\Users\Welcome\Machine Learning\Daily Tasks\model_drug1","wb"))


# In[39]:


#lb.classes_


# In[41]:




# In[44]:





# In[ ]:


pre = pickle.load(open(r"C:\Users\Welcome\Machine Learning\Daily Tasks\final_drug1","rb"))

model1 = pickle.load(open(r"C:\Users\Welcome\Machine Learning\Daily Tasks\model_drug1","rb"))

s.title("Pridict The Drug")
s.write("Maheswari")
age = s.number_input("Enter the age",min_value=0,max_value=100)
gender = s.radio("select the gender",["F","M"])
Bp = s.radio("select BP of a person",["HIGH","LOW","NORMAL"])
chelostral = s.radio("select the chelostral of a person",["HIGH","NORMAL"])
na_to_k = s.number_input("Enter the Value")


query1 = pre.transform(pd.DataFrame([[age,gender,Bp,chelostral,na_to_k]]
                                       ,columns=["Age","Sex","BP","Cholesterol","Na_to_K"]))
pre1 = model1.predict(query1)

if pre1 == 0:
    x = "DrugY"
elif pre1 == 1:
    x = "drugA"
elif pre1 == 2:
    x = "drugB"
elif pre1 == 3:
    x = "drugC"
else:
    x = "drugX"


if s.button("submit"):
    s.write(x)

