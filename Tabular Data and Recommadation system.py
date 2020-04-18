#!/usr/bin/env python
# coding: utf-8

# # Handling tabular data

# In[1]:


from fastai.collab import *
from fastai.tabular import *


# In[2]:


#here you unzip the data
path=untar_data(URLs.ADULT_SAMPLE)


# In[3]:


#listing the file contained in Data(mk-100k(data folder))
path.ls()


# In[12]:


#simply read the csv file, path is the location of dataset
df=pd.read_csv(path/'adult.csv')


# In[13]:


#print the first 5 datas
df.head()


# In[14]:


"""now as in the above dataframe (tabular data) you can see that there
are NaN which denotes null value so you have to immute(remove or replace)
it.So for that you need know the features with null value."""
df.isnull().sum()


# In[15]:


#once ypu know which features have null value check the Dtype .
df.info()


# In[18]:


"""In deep-learning just like ml we have target value which is known as 
Dependent value.here it is salary"""
dep_var = 'salary'
#now seperate the catagorical variables.
cat_name =['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
cont_names=['age','fnlwgt','education-num']


# In[19]:


#
procs = [FillMissing,Categorify,Normalize]


# In[21]:


#testing data is ready

test = TabularList.from_df(df.iloc[800:1000].copy(),path=path,cat_names=cat_name,cont_names=cont_names)


# In[32]:


#now here data is actually training data


data = (TabularList.from_df(df,path=path,cat_names=cat_name,cont_names=cont_names,procs=procs)
                           .split_by_idx(list(range(800,1000))) 
                           .label_from_df(col))


# In[33]:


data


# In[ ]:


model_tabular = 

