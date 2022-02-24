#%%
from datetime import datetime
from re import T
import pandas as pd
import numpy as np
import seaborn as sns
#%%
plants_df = pd.read_excel('../dataset/Data.xlsx', sheet_name='plants')
flights_df = pd.read_excel('../dataset/Data.xlsx', sheet_name='flight dates')
planting_df = pd.read_excel('../dataset/Data.xlsx', sheet_name='planting')
planting_df = planting_df.iloc[:1822 , :]
weather_df = pd.read_excel('../dataset/Data.xlsx', sheet_name='weather')
# %%
df = plants_df.merge(flights_df, on='Batch Number', how='inner').reset_index()
# df.to_csv('../dataset/plants_flights1.csv')
#%%
planting_df.rename(columns={"Plant_Date": "Plant Date"}, inplace=True)
# %%
planting_df['Plant Date'] = pd.to_datetime(planting_df['Plant Date'], errors = 'coerce')
df['Plant Date'] = pd.to_datetime(df['Plant Date'], errors = 'coerce')
#%%
df = df.merge(planting_df, on = 'Plant Date', how='inner').reset_index(drop=True)
df.drop(columns=['Column2', 'Column1', 'Column3', 'Column4'], inplace = True)
# df.to_csv('../dataset/plants_flights2.csv')
# %%
df_og = pd.read_csv('../dataset/plants_flights2.csv')
df_og.describe()
#%%
df = df_og
df.columns
df.describe()

# %%
df.info
#%%
# df = df[df['Remove'].map(len) >= 1]
# df = df[df['Remove'].isna()]
# %%
df.describe()


# %%
sns.pairplot(weather_df)
#%%
# %%
weather_df_og = weather_df
weather_df.drop(columns=['Wind Speed [max]', 'Battery Voltage [last]', 
'Air Temperature [max]', 'Air Temperature [min]', 'Dew Point [min]'], inplace= True)
weather_df.columns

# %%
# weather_df['dates'] = pd.DatetimeIndex(weather_df['dates'])
# weather_df.info()
#%%
weather_df['dates'] = weather_df['Unnamed: 0'].astype(object)
weather_df['dates'] = pd.to_datetime(weather_df['dates'])

weather_df.info()

#%%
weather_df['day'] = weather_df['dates'].dt.day
weather_df['month'] = weather_df['dates'].dt.month
weather_df['year'] = weather_df['dates'].dt.year

# %%
weather_df.info()
# %%
weather_df.head()
# %%
weather_df_2014 = weather_df[weather_df['year'] == 2014]
weather_df_2015 = weather_df[weather_df['year'] == 2015]
weather_df_2016 = weather_df[weather_df['year'] == 2016]
weather_df_2017 = weather_df[weather_df['year'] == 2017]
weather_df_2018 = weather_df[weather_df['year'] == 2018]
weather_df_2019 = weather_df[weather_df['year'] == 2019]
# %%
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
# %%
weather_df_2014.drop(columns=['year', 'Unnamed: 0', 'dates', 'ET0 [result]'], inplace=True)
weather_df_2015.drop(columns=['year', 'Unnamed: 0', 'dates', 'ET0 [result]'], inplace=True)
weather_df_2016.drop(columns=['year', 'Unnamed: 0', 'dates', 'ET0 [result]'], inplace=True)
weather_df_2017.drop(columns=['year', 'Unnamed: 0', 'dates', 'ET0 [result]'], inplace=True)
weather_df_2018.drop(columns=['year', 'Unnamed: 0', 'dates', 'ET0 [result]'], inplace=True)
weather_df_2019.drop(columns=['year', 'Unnamed: 0', 'dates', 'ET0 [result]'], inplace=True)
# %%
weather_df_2014.name = 'wet_2014'
weather_df_2015.name = 'wet_2015'
weather_df_2016.name = 'wet_2016'
weather_df_2017.name = 'wet_2017'
weather_df_2018.name = 'wet_2018'
weather_df_2019.name = 'wet_2019'
#%%
weather_df_2019.describe()
#%%
# weather_df_2019.isnull().values.any()
weather_df_2018.info()
#%%
weather_df_2018.dropna(inplace=True)

#%%
weather_df_2018.info()
#%%
sns.pairplot(weather_df_2019)
#%%
wet_dfs = [
weather_df_2014,
weather_df_2015,
weather_df_2016,
weather_df_2017,
weather_df_2018,
weather_df_2019
]
x_list = []
y_list = []
model_list = []
for i in wet_dfs:
    print(i.name)
    tmp_var = str(i.name)
    i.replace([np.inf, -np.inf], np.nan, inplace=True)
    # np.all(np.isfinite(i))
    i.dropna(inplace=True)
    i = i.reset_index()
    tmp_var_x = 'x_'+tmp_var
    tmp_var_y = 'y_'+tmp_var
    tmp_var_y = i.iloc[:,:-2]
    tmp_var_x = i.iloc[:,-2:]
    reg_var = 'model_' + tmp_var
    reg_var = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(tmp_var_x,tmp_var_y)
    x_list.append(tmp_var_x)
    y_list.append(tmp_var_y)
    model_list.append(reg_var)
    # print(reg_var)
#%%
print(x_list)
#%%
y = weather_df_2018.iloc[:,:-2]
x = weather_df_2018.iloc[:,-2:]
#%%
y.iloc[10]
#%%
y.columns
# %%
wet_2018 = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(x,y)
#%%
wet_2018.predict([[24,5]])
# %%
print(model_list)

# %%
