#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression , Lasso , Ridge , LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Cell 2 contains influence of features on target

# Cell 3 contains our models predictions of wildfires


# In[34]:


# Getting whole wildfire data set
wildfire_df = pd.read_csv("2- annual-area-burnt-by-wildfires.csv")

# Filtering wildfire data set for just 2023 as our features are for 2023
wildfire_df = wildfire_df[wildfire_df['Year'] == 2023]

wildfire_df = wildfire_df.drop("Year" , axis = 1)

# Getting whole climate data 
climate_df = pd.read_csv("world-data-2023.csv")

# Making sure the number of countries 
wildfire_df = wildfire_df[wildfire_df['Country'].isin(climate_df['Country'])]
                    
# Adding the features to one data frame 
merged_df = pd.merge(wildfire_df, climate_df, on='Country')

# Filling missing values with the mean 
merged_df = merged_df.fillna(merged_df.mean())

#merged_df = merged_df[merged_df['Annual area burnt by wildfires'] > 0] 

X = merged_df.drop(["Country" , "Annual area burnt by wildfires"] , axis = 1)

y = merged_df["Annual area burnt by wildfires"]


X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 1)

lasso = Lasso(alpha=0.3 , normalize = True)
lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_

y_pred = lasso.predict(X_test)

plt.figure(figsize=(12, 8))

# Horizontal bar plot
plt.barh(X.columns, lasso_coef, color='teal')

# Add labels and title
plt.xlabel('Coefficient Value', fontsize=14, labelpad=10)
plt.ylabel('Features', fontsize=14, labelpad=10)
plt.title('Lasso Regression Coefficients', fontsize=16, pad=15)

# Optional: Add gridlines for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Rotate the feature names for better readability (optional)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()


# In[35]:


lasso = Ridge(alpha=0.3 , normalize = True)
lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_

y_pred = lasso.predict(X_test)

plt.figure(figsize=(12, 8))

# Horizontal bar plot
plt.barh(X.columns, lasso_coef, color='teal')

# Add labels and title
plt.xlabel('Coefficient Value', fontsize=14, labelpad=10)
plt.ylabel('Features', fontsize=14, labelpad=10)
plt.title('Lasso Regression Coefficients', fontsize=16, pad=15)

# Optional: Add gridlines for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Rotate the feature names for better readability (optional)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:


X = merged_df["Forested Area (%)"].values.reshape(-1,1)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 42)

reg = Lasso(alpha = 0.3)


reg.fit(X_train , y_train)

y_pred = reg.predict(X_test)

plt.scatter(X , y)

plt.plot(X_test , y_pred , color = 'r')

plt.xlabel("Forested Area")
plt.ylabel("Area Burnt")


plt.show()

