#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[39]:


df = pd.read_csv(r"C:\Users\Arun\OneDrive\Desktop\carprice\Unemployment in India.csv")
df.columns = df.columns.str.strip() # Removing starting and ending spaces from column names
df


# In[40]:


df.columns # Checking column names


# In[41]:


# Checking for null values
df['Region'].value_counts(dropna=False)


# In[42]:


df['Region'].isnull().sum()


# In[43]:


nan_index_Region = df[df['Region'].isnull()].index.tolist()


# In[44]:


# Checking for null values
df['Date'].value_counts(dropna=False)


# In[45]:


# Checking for null values
nan_index_Date = df[df['Date'].isnull()].index.tolist()


# In[46]:


df['Frequency'] = df['Frequency'].str.replace(' Monthly', 'Monthly')

# Checking for null values
df['Frequency'].value_counts(dropna=False)


# In[47]:


# Checking for null values
nan_index_Frequency = df[df['Frequency'].isnull()].index.tolist()


# In[48]:


# Checking for null values
df['Estimated Unemployment Rate (%)'].value_counts(dropna=False)


# In[49]:


# Checking for null values
nan_index_Unemployment_Rate = df[df['Estimated Unemployment Rate (%)'].isnull()].index.tolist()


# In[50]:


# Checking for null values
df['Estimated Employed'].value_counts(dropna=False)


# In[51]:


# Checking for null values
nan_index_Employed = df[df['Estimated Employed'].isnull()].index.tolist()


# In[52]:


# Checking for null values
df['Estimated Labour Participation Rate (%)'].value_counts(dropna=False)


# In[53]:


# Checking for null values
nan_index_Labour = df[df['Estimated Labour Participation Rate (%)'].isnull()].index.tolist()


# In[54]:


# Checking for null values
df['Area'].value_counts(dropna=False)


# In[55]:


# Checking for null values
nan_index_Area = df[df['Area'].isnull()].index.tolist()


# In[56]:


df = df.dropna()
df


# In[57]:


Employed = df['Estimated Employed'].iloc[:50]
Unemployed = df['Estimated Unemployment Rate (%)'].iloc[:50]

#normalize the employed in range 1 to 20
Employed = (Employed - Employed.min())/(Employed.max() - Employed.min()) * 19 + 1

plt.figure(figsize=(12,10))
plt.plot(Employed, label='Employed', color='blue')
plt.plot(Unemployed, label='Unemployed', color='red')
plt.show


# In[58]:


#get correlation of df['Estimated Employed'] and  df['Estimated Unemployment Rate (%)']
print(df['Estimated Employed'].corr(df['Estimated Unemployment Rate (%)']))


# In[59]:


df = df.drop(['Estimated Employed'], axis=1)
df


# In[60]:


#Break down the column Date on basis of - i.e. into day, month, year
df[['Day','Month','Year']] = df.Date.str.split("-",expand=True,)
df = df.drop(['Date'], axis=1)
df


# In[61]:


group = df.groupby(['Region','Area']).agg({'Estimated Unemployment Rate (%)': 'mean'})
group


# In[ ]:


# Unemployement rate was more in the urban areas of the states/region

