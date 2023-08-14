#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[88]:



# Load the dataset
data_path = r"C:\Users\Arun\OneDrive\Desktop\carprice\CarPrice_Assignment.csv"
df = pd.read_csv(data_path)
df


# In[89]:



# Preprocess the data
# Drop any rows with missing values
df.dropna(inplace=True)


# In[90]:



# Extract 'make' from 'CarName' and drop 'CarName' column
df['make'] = df['CarName'].apply(lambda x: x.split(' ')[0])
df.drop(['car_ID', 'CarName'], axis=1, inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['make', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                    'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# In[91]:



# Splitting data into features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[92]:



# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[93]:



# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predicting on the test set
y_pred = model.predict(X_test_scaled)

# Calculate accuracy metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("accuracy:", r2)


# In[94]:



# Visualize the predicted vs. actual prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Car Prices")
plt.show()


# In[97]:


# Plot a heatmap of the correlation matrix
plt.figure(figsize=(20, 14))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:




