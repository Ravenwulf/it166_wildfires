#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression , Lasso , Ridge , LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df1 = pd.read_csv("1- cumulative-area-burnt-by-wildfires-by-week.csv").drop("2024" , axis = 1)
df2 = pd.read_csv("2- annual-area-burnt-by-wildfires.csv")
df3 = pd.read_csv("3- share-of-the-total-land-area-burnt-by-wildfires-each-year.csv")
df4 = pd.read_csv("4- annual-area-burnt-per-wildfire.csv")
df5 = pd.read_csv("5- annual-burned-area-by-landcover.csv")

cross_df = pd.read_csv("climate_change_data.csv")
#                                                 Notes on Project

# We need to find out what the units are for this data set.

# Pearsons Correlation coefficient for each feature relative to target using Sci py

# AJupyter lab interface over Jupyer notebook

# Videos of explaining the problem. Explain with a voiceover while having a video of a fire
        


# In[2]:


df2 = df2.drop(df2[df2['Year'] == 2024].index)

X = df2["Year"].values
y = df2["Annual area burnt by wildfires"].values


# Plotting on the first subplot
plt.bar(X, y)  
plt.title("Annual Area Burnt by Wildfires")
plt.xlabel("Year")
plt.ylabel("Area Burnt")

plt.show()
# Wildfire impact has remain relatively stable over the years


# In[3]:


climate_df = pd.read_csv("climate_change_data.csv")

list_of_countries = ["Brazil", "United States of America", "France", "China", "Egypt", "Australia"]

# Filter the DataFrame using the 'isin()' method
filtered_df = climate_df[climate_df["Country"].isin(list_of_countries)]

climate_features = ["Temperature" , "CO2 Emissions" , "Sea Level Rise" , "Precipitation" , "Humidity" , "Wind Speed"]

mean_per_country = filtered_df.groupby("Country")[climate_features].mean()

mean_per_country # Calculates the mean of each feature in the given country to give us an idea of how each relates to one another


# In[4]:


wildfire_df = pd.read_csv("1- cumulative-area-burnt-by-wildfires-by-week.csv")

wildfire_df = wildfire_df.drop("2024" , axis = 1)

list_of_countries = ["Brazil", "United States", "France", "China", "Egypt", "Australia"]

filtered_wildfire_df = wildfire_df[wildfire_df["Entity"].isin(list_of_countries)]

list_of_cumulation = ["2023" , "2022" , "2021" , "2020" , "2019" , "2018" , "2017" , "2016" , "2015" , "2014" , "2013" , "2012"]

wildfire_mean = filtered_wildfire_df.groupby("Entity")[list_of_cumulation].mean()

wildfire_mean # Grouped average of the cumulative area burnt (yearly) of every country in every year


# In[5]:


# Based on the data points above, we can see that each country experiences similar wildfires within their domain

# This cell will focus on establishing the most useful features and filtering the data to the best of our ability

import seaborn as sns
import matplotlib.pyplot as plt


# Checking the significance of each feature relative to the target in this case 2023 as all of the years are similar
for feature in climate_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=climate_df[feature], y=wildfire_df['2023']) # Using sns here because matplotlib would not work
    plt.title(f'Relationship between {feature} and Target')
    plt.xlabel(feature)
    plt.ylabel('2023 Cumulative Area Burnt')
    plt.show()
    
# The significance is not showing to be extremely important,  WE NEED TO INVESTIGATE WHY. It appears that it is only
# somewhat relevant for the worst wildfires


# In[ ]:





# In[6]:


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
        reg = Lasso(alpha=.2)  
    elif model == "Ridge":
        reg = Ridge(alpha=.1)  
    
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




