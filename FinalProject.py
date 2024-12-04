#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


# Getting whole wildfire data set
wildfire_df = pd.read_csv("share-of-the-total-land-area-burnt-by-wildfires-each-year.csv")

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
#print(merged_df.head())
merged_df = merged_df.drop("Country", axis=1)
merged_df = merged_df.fillna(merged_df.mean())

scaler = StandardScaler();
normalized = pd.DataFrame(
    scaler.fit_transform(merged_df),
    columns=merged_df.columns,
    index=merged_df.index
)

merged_df = normalized

#merged_df = merged_df[merged_df['Annual share of the total land area burnt by wildfires'] > 0]

#merged_df = merged_df[merged_df['Annual area burnt by wildfires'] > 0]
#merged_df = merged_df[merged_df['Annual area burnt by wildfires'] < 200000]
#merged_df = merged_df[merged_df["CO2 Emissions"] < 300000]

# X = merged_df.drop("Annual area burnt by wildfires", axis = 1)

# y = merged_df["Annual area burnt by wildfires"]

X = merged_df.drop("Annual share of the total land area burnt by wildfires", axis = 1)

y = merged_df["Annual share of the total land area burnt by wildfires"]


X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 1)

lasso = Lasso(alpha=0.3)
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


ridge = Ridge(alpha=0.3)
ridge.fit(X_train, y_train)
ridge_coef = ridge.coef_

y_pred = ridge.predict(X_test)

plt.figure(figsize=(12, 8))

# Horizontal bar plot
plt.barh(X.columns, ridge_coef, color='teal')

# Add labels and title
plt.xlabel('Coefficient Value', fontsize=14, labelpad=10)
plt.ylabel('Features', fontsize=14, labelpad=10)
plt.title('Ridge Regression Coefficients', fontsize=16, pad=15)

# Optional: Add gridlines for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Rotate the feature names for better readability (optional)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

# Prepare the features (X) and target (y)
X = merged_df["Birth Rate"].values.reshape(-1, 1)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the Ridge regression model
reg1 = Ridge(alpha=0.3)
reg1.fit(X_train, y_train)
y_pred1 = reg1.predict(X_test)

# Initialize and fit the Lasso regression model
reg2 = Lasso(alpha=0.3)
reg2.fit(X_train, y_train)
y_pred2 = reg2.predict(X_test)

# Set up the figure size and style
plt.figure(figsize=(12, 8))  # Increase size of the plot
plt.style.use('seaborn-white')  # Use seaborn's clean whitegrid style

# Scatter plot of the actual data points
plt.scatter(X, y, color='blue', alpha=0.6, label="Data Points", edgecolors="black", s=100)

# Plot the Ridge and Lasso regression lines
plt.plot(X_test, y_pred1, color='red', label="Ridge Regression (alpha=0.3)", linewidth=2)
plt.plot(X_test, y_pred2, color='green', label="Lasso Regression (alpha=0.3)", linewidth=2)

# Add labels and title
plt.title("Regression Comparison: Ridge vs Lasso", fontsize=16)
plt.xlabel("Birth Rate", fontsize=14)
plt.ylabel("Area Burnt", fontsize=14)

# Add grid, legend, and improve layout
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12, loc='best')
plt.tight_layout()

# Show the plot
plt.show()

# Scatter plot for Ridge regression: Actual vs Predicted
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred1, color='red', edgecolors='black', alpha=0.7, s=100)
plt.title("Ridge Regression: Actual vs Predicted", fontsize=16)
plt.xlabel("Actual Area Burnt", fontsize=14)
plt.ylabel("Predicted Area Burnt (Ridge)", fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Show the second plot
plt.show()

# Print the R-squared scores for both models
print(f"R-squared (Ridge): {reg1.score(X_test, y_test):.3f}")
print(f"R-squared (Lasso): {reg2.score(X_test, y_test):.3f}")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors


# Calculate the Pearson correlation between each feature and the target variable
correlations_with_target = merged_df.corr()[target_column].drop(target_column)

# Making a custom colored heatmap for readability
blue_to_red = mcolors.LinearSegmentedColormap.from_list(
    "blue_to_red", ["red", "white", "blue"], N=256
)

# Set up the figure size for the heatmap
plt.figure(figsize=(18, 14))

sns.heatmap(
    correlations_with_target.to_frame(),  
    annot=True,                          
    vmin=-1, vmax=1,                     
    cmap=blue_to_red, 
    annot_kws={"size": 20} ,
)

# Getting the scale bar object so we can adjust the size
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=20) 

plt.yticks(fontsize=20)  
plt.xticks(fontsize=20)  


plt.title('Features and Area Burnt', fontsize=25)
plt.xlabel('Target: Share of land burned' , fontsize=25)
plt.ylabel('Features', fontsize=25)

plt.tight_layout()

plt.show()


# In[ ]:




