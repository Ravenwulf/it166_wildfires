# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("imtkaggleteam/wildfires")

# print("Path to dataset files:", path)
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.linear_model import LinearRegression;

data_tables = ["cumulative-area-burnt-by-wildfires-by-week.csv", "annual-area-burnt-per-wildfire.csv", "annual-area-burnt-by-wildfires.csv", "annual-burned-area-by-landcover.csv", "share-of-the-total-land-area-burnt-by-wildfires-each-year.csv"]

# for table in data_tables:
#     df = pd.read_csv(table)
#     print(df.info)
#     print(df.columns.values)

# CUM_AREA_BURNT = pd.read_csv("cumulative-area-burnt-by-wildfires-by-week.csv")
# print(CUM_AREA_BURNT.columns.values)

aabpw = pd.read_csv(data_tables[1]);
aabpw = aabpw.drop(aabpw[aabpw["Year"] == 2024].index);

ababl = pd.read_csv(data_tables[3]);
ababl = ababl.drop(aabpw[aabpw["Year"] == 2024].index);


ababl_groups = ababl.groupby(["Year", "Code"])["Yearly burned area across croplands"].sum();

print(ababl_groups);




# print(aabpw.head());

# aabpw_big = aabpw.drop(aabpw[aabpw["Annual area burnt per wildfire"] < 7000].index);

# print(aabpw_big.head());

# y = aabpw_big["Annual area burnt per wildfire"].values;
# X = aabpw_big["Year"].values;

# plt.scatter(X, y)
# plt.ylabel("Annual area burnt per wildfire greater than 7000 km^2 (km^2)")
# plt.xlabel("Year")
# plt.show()

# aabpw_mean_by_yr = aabpw.groupby('Year')['Annual area burnt per wildfire'].mean()

# print(aabpw_mean_by_yr);


# y = aabpw_mean_by_yr.index;
# X = aabpw_mean_by_yr.values;
# print(y);
# print(X);

# plt.plot(y, X)
# plt.xlabel("Year")
# plt.ylabel("Avg area burnt")
# plt.show();

