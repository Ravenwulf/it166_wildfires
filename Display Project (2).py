#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install plotly')


import pandas as pd
import plotly.express as px

wildfire_df = pd.read_csv("2- annual-area-burnt-by-wildfires.csv")
wildfire_df = wildfire_df[wildfire_df["Year"] == 2023]
climate_df = pd.read_csv("world-data-2023.csv")
wildfire_df = wildfire_df[wildfire_df['Country'].isin(climate_df['Country'])]

data = {
    'Country': list(climate_df["Country"].unique()),
    'Wildfire_Damage': list(wildfire_df["Annual area burnt by wildfires"])  
}

df = pd.DataFrame(data)

fig = px.choropleth(df, 
                    locations="Country", 
                    locationmode="country names",  
                    color="Wildfire_Damage",  
                    hover_name="Country",  
                    labels={"Wildfire_Damage": "Annual Area Burnt"},
                    title="2023 Total Area Burnt by Country (KM^2)")


fig.show()


# In[3]:


# Getting whole wildfire data set
wildfire_df = pd.read_csv("share-of-the-total-land-area-burnt-by-wildfires-each-year.csv")

# Filtering wildfire data set for just 2023 as our features are for 2023
wildfire_df = wildfire_df[wildfire_df['Year'] == 2023]

# Dropping the 'Year' column
wildfire_df = wildfire_df.drop("Year", axis=1)

# Getting whole climate data
climate_df = pd.read_csv("world-data-2023.csv")

# Making sure the number of countries match between both datasets
wildfire_df = wildfire_df[wildfire_df['Country'].isin(climate_df['Country'])]

# Merging the two datasets on 'Country'
merged_df = pd.merge(wildfire_df, climate_df, on='Country')


merged_df['Wildfire Damage'] = merged_df['Annual share of the total land area burnt by wildfires']  # Replace 'Area Burnt' with actual column name if needed

data = {
    'Country': merged_df["Country"],
    'Wildfire Damage': merged_df["Wildfire Damage"],  # Add 'Wildfire Damage' here
    
}

df = pd.DataFrame(data)

fig = px.choropleth(df, 
                    locations="Country", 
                    locationmode="country names",  
                    color="Wildfire Damage", 
                    hover_name="Country",  
                    color_continuous_scale="Viridis",  
                    labels={"Wildfire Damage": "Annual Area Burnt (%)"},  
                    title="2023 Total Percentage of Area Burnt by Country")

fig.show()


# In[ ]:




