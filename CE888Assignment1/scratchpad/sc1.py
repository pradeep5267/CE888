#%%
from datetime import datetime
import pandas as pd
import numpy as np

#%%
plants_df = pd.read_excel('../dataset/Data.xlsx', sheet_name='plants')
flights_df = pd.read_excel('../dataset/Data.xlsx', sheet_name='flight dates')
planting_df = pd.read_excel('../dataset/Data.xlsx', sheet_name='planting')
weather_df = pd.read_excel('../dataset/Data.xlsx', sheet_name='weather')


# %%
weather_df.describe()
#%%
print(plants_df.columns)
print(flights_df.columns)
print(planting_df.columns)
print(weather_df.columns)

# %%
df = plants_df.merge(flights_df, on='Batch Number', how='inner').reset_index()
df.describe()
# %%
df.to_csv('../dataset/plants_flights.csv')
# %%
weather_df.rename(columns={"Unnamed: 0": "Check Date"}, inplace=True)
df_wet = df.merge(weather_df, on = 'Check Date', how='inner').reset_index()
df_wet.describe()
# %%
# planting_df.rename(columns={"Plant_Date": "Plant Date"}, inplace=True)
# datetime.datetime.utcfromtimestamp(int(x)/1e9)
# planting_df['Plant Date'] = planting_df['Plant Date'].astype(np.datetime64)
df_wet['Plant Date'] = pd.to_datetime(df_wet['Plant Date'], errors = 'coerce')
#%%
df_wet.drop('Unnamed: 0',inplace=True)
planting_df.drop('Unnamed: 0',inplace=True)

#%%
df_wet_pat = df_wet.merge(planting_df, on = 'Plant Date', how='inner').reset_index(drop=True)
df_wet.describe()

# %%
df_wet_pat.to_csv('../dataset/weather_planting.csv')
#%%
planting_df.dtypes
#%%
weather_df.dtypes
#%%
flights_df.dtypes
#%%
plants_df.dtypes