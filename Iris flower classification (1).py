#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np                     
import pandas as pd                    
import seaborn as sns
import matplotlib.pyplot as plt       
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
iris = pd.read_csv(r"C:\Users\Arun\Downloads\irisdata.csv")
iris.head() 


# In[2]:


iris.info()


# In[3]:


iris.describe()


# In[4]:


sns.boxplot(x="sepal_length",y="species",data=iris)


# In[5]:


boxp = sns.stripplot(x="species", y="petal_length", data=iris, jitter=True, edgecolor="gray")


# In[6]:


plot=iris[iris.species=="Iris-setosa"].plot.scatter(x="petal_length",y="petal_width",color='b',label="Iris-setosa")
iris[iris.species=="Iris-versicolor"].plot.scatter(x="petal_length",y="petal_width",color='g',label="Iris-versicolor",ax=plot)
iris[iris.species=="Iris-virginica"].plot.scatter(x="petal_length",y="petal_width",color='r',label="Iris-virginica",ax=plot)

plt.title("IRIS : petal_length Vs petal_width")
plt.show()


# In[7]:


plot=iris[iris.species=="Iris-setosa"].plot.scatter(x="sepal_length",y="petal_length",color='b',label="Iris-setosa")
iris[iris.species=="Iris-versicolor"].plot.scatter(x="sepal_length",y="petal_length",color='g',label="Iris-versicolor",ax=plot)
iris[iris.species=="Iris-virginica"].plot.scatter(x="sepal_length",y="petal_length",color='r',label="Iris-virginica",ax=plot)

plt.title("IRIS : sepal_length Vs petal_length")
plt.show()


# In[8]:


plot=iris[iris.species=="Iris-setosa"].plot.scatter(x="sepal_width",y="petal_width",color='b',label="Iris-setosa")
iris[iris.species=="Iris-versicolor"].plot.scatter(x="sepal_width",y="petal_width",color='g',label="Iris-versicolor",ax=plot)
iris[iris.species=="Iris-virginica"].plot.scatter(x="sepal_width",y="petal_width",color='r',label="Iris-virginica",ax=plot)

plt.title("IRIS : sepal_width Vs petal_width")
plt.show()


# In[9]:


sns.set_theme(style="ticks")

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")


# In[10]:


iris.hist(edgecolor='black',figsize=(10,10))
plt.show()


# In[11]:


from sklearn import metrics
from sklearn.metrics import accuracy_score
x = iris.drop('species', axis=1)
y= iris.species
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)
logreg = LogisticRegression()
logreg.fit(x, y)
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x)


# In[12]:


print(metrics.accuracy_score(y, y_pred))


# In[13]:


logreg.score(x_test,y_test)


# In[14]:


logreg.score(x_train,y_train)


# In[15]:


from sklearn.metrics import classification_report
from sklearn.svm import SVC


# In[16]:


svm = SVC(gamma='auto')
svm.fit(x_train, y_train)
prediction = svm.predict(x_test)
print(f'Accuracy of SVM: {accuracy_score(y_test, prediction)}')
print(f'Report of SVM: \n {classification_report(y_test, prediction)}')


# In[ ]:




