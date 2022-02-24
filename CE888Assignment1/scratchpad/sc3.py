#%%
import pandas as pd
import numpy as np
from datetime import datetime


#%%
weather_df = pd.read_excel('../dataset/Data.xlsx', sheet_name='weather')
weather_df.describe()
#%%
weather_df.info()
#%%
weather_df.head()
#%%
# weather_df['Unnamed: 0'].astype(datetime)
weather_df['Unnamed: 0'] = pd.to_datetime(weather_df['Unnamed: 0'], format="%Y%m%d%H%M%S")
#%%
weather_df.head()

#%%
weather_df['DATE'] = pd.to_datetime(weather_df['Unnamed: 0'])

# %%
weather_df2 = weather_df[(weather_df['DATE'].dt.month == 5) & (weather_df['DATE'].dt.year == 2014)]

# %%
weather_df2.head()
#%%
weather_df2.info()
# %%
for i in range():
