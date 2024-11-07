# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("imtkaggleteam/wildfires")

# print("Path to dataset files:", path)

import pandas as pd

data_tables = ["cumulative-area-burnt-by-wildfires-by-week.csv", "annual-area-burnt-per-wildfire.csv", "annual-area-burnt-by-wildfires.csv", "annual-burned-area-by-landcover.csv", "share-of-the-total-land-area-burnt-by-wildfires-each-year.csv"]

for table in data_tables:
    df = pd.read_csv(table)
    print(df.info)
    print(df.columns.values)

# CUM_AREA_BURNT = pd.read_csv("cumulative-area-burnt-by-wildfires-by-week.csv")
# print(CUM_AREA_BURNT.columns.values)

