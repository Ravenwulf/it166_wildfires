#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression , Lasso , Ridge , LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#                                                 Notes on Project

# Pearsons Correlation coefficient for each feature relative to target using Sci py


# In[20]:


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

# percentage_features = [
# 'Agricultural Land( %)' , 
# 'CPI Change (%)' , 
# 'Forested Area (%)' , 
# 'Gross primary education enrollment (%)' ,
# 'Gross tertiary education enrollment (%)',
# 'Out of pocket health expenditure' ,
# 'Population: Labor force participation (%)' , 
# 'Tax revenue (%)' , 
# 'Total tax rate' ,
# 'Unemployment rate' ,
# ]

# # Converting all percentages to decimal places and removing the symbol to convert it to a float
# for item in percentage_features:
#     merged_df[item] = merged_df[item].str.replace('%', '').astype(float) / 100

merged_df = merged_df.fillna(merged_df.mean())


X = merged_df.drop(["Country" , "Annual area burnt by wildfires"] , axis = 1)

y = merged_df["Annual area burnt by wildfires"]

reg = LinearRegression()

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 42)

reg.fit(X_train , y_train)

y_pred = reg.predict(X_test)

plt.scatter(X['CO2 Emissions'] , y)

plt.plot(X_test["CO2 Emissions"] , y_pred)
# print(y_pred)

# print()

# print(y_test)
# display(merged_df.iloc[45])
print(y_pred[:2])
display(y_test[:2])


# In[ ]:



# climate_countries = ["Brazil", "United States of America", "France", "China", "Egypt", "Australia"]

# # Filter the DataFrame using the 'isin()' method
# filtered_climate_df = climate_df[climate_df["Country"].isin(climate_countries)]

# climate_features = ["Temperature" , "CO2 Emissions" , "Sea Level Rise" , "Precipitation" , "Humidity" , "Wind Speed"]

# mean_per_country = filtered_climate_df.groupby("Country")[climate_features].mean()

# mean_per_country # Calculates the mean of each feature in the given country to give us an idea of how each relates to one another

climate_df.shape


# In[ ]:


wildfire_countries = ["Brazil", "United States", "France", "China", "Egypt", "Australia"]

filtered_wildfire_df = wildfire_df[wildfire_df["Entity"].isin(wildfire_countries)]

area_burnt = ["2023" , "2022" , "2021" , "2020" , "2019" , "2018" , "2017" , "2016" , "2015" , "2014" , "2013" , "2012"]

wildfire_mean = filtered_wildfire_df.groupby("Entity")[area_burnt].mean()

wildfire_mean # Grouped average of the cumulative area burnt (yearly) of every country in every year


# In[ ]:


# Based on the data points above, we can see that each country experiences similar wildfires within their domain

# This cell will focus on establishing the most useful features and filtering the data to the best of our ability

import seaborn as sns
import matplotlib.pyplot as plt


# Checking the significance of each feature relative to the target in this case 2023 as all of the years are similar
# for feature in climate_features:
#     plt.figure(figsize=(8, 5))
#     sns.scatterplot(x=filtered_climate_df[feature], y=filtered_wildfire_df['2023']) # Using sns here because matplotlib would not work
#     plt.title(f'Relationship between {feature} and Target')
#     plt.xlabel(feature)
#     plt.ylabel('2023 Cumulative Area Burnt')
#     plt.show()

sns.scatterplot(x = filtered_climate_df['CO2 Emissions'] , y = filtered_wildfire_df['2023'])

filtered_wildfire_df


# In[ ]:



print(filtered_df[filtered_df["Country"] == 'Brazil'].shape)
print(filtered_df[filtered_df["Country"] == 'United States of America'].shape)
print(filtered_df[filtered_df["Country"] == 'France'].shape)
print(filtered_df[filtered_df["Country"] == 'China'].shape)
print(filtered_df[filtered_df["Country"] == 'Egypt'].shape)
print(filtered_df[filtered_df["Country"] == 'Australia'].shape)


# In[ ]:


# We can play with the formatting, I think we need to evaluate our data for faults as our metric scores are awful.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# List of models to test
list_of_models = ["Linear", "Lasso", "Ridge"]


# Function to create model and plot predictions, and also return accuracy metrics
def model_maker(model, X_train, X_test, y_train, y_test, ax, model_idx):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Selecitng the appropriate model at this point in the function/loop
    if model == "Linear":
        reg = LinearRegression()
    elif model == "Lasso":
        reg = Lasso(alpha=100000)  
    elif model == "Ridge":
        reg = Ridge(alpha=1)  
    
    # Fit the model
    reg.fit(X_train_scaled, y_train)
    
    # Predicting the values
    y_pred = reg.predict(X_test_scaled)
    
    # Plot the scatter plot in the correct axis
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Sample Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{model} Model")
    
    # Testing all of our accuracy metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    # Return the metrics
    return r2, mae, mse, rmse

# Loop through each wildfire year and create the subplots for all models
for wildfire in list_of_cumulation:
    X = climate_df[climate_features]
    y = wildfire_df[wildfire]
    
    # Changing sample to 10,000 rows to match our features
    y = y.sample(n=10000, random_state=41)
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))  
    
    # Making a dictionary to store our values for printing later
    metrics = {}

    # Loop through the models and create a subplot for each model
    for idx, model in enumerate(list_of_models):
        r2, mae, mse, rmse = model_maker(model, X_train, X_test, y_train, y_test, ax[idx], idx)
        metrics[model] = {
            "R²": r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        }
    
    # Title for the given year
    fig.suptitle(f"Regression Models for {wildfire}", fontsize=16)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  

    # Show the plot for this year
    plt.show()

    # Printing the accuracy of each model
    print(f"Metrics for {wildfire} year:")
    for model in list_of_models:
        print(f"\n{model} Model:")
        print(f"\tR²: {metrics[model]['R²']:.4f}")
        print(f"\tMAE: {metrics[model]['MAE']:.4f}")
        print(f"\tMSE: {metrics[model]['MSE']:.4f}")
        print(f"\tRMSE: {metrics[model]['RMSE']:.4f}")


# In[ ]:




