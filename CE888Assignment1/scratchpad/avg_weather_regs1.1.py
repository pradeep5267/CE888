#%%
from datetime import datetime
from re import T
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
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
planting_df['Plant Date'] = pd.to_datetime(planting_df['Plant Date'], errors = 'coerce')
df['Plant Date'] = pd.to_datetime(df['Plant Date'], errors = 'coerce')
#%%
df = df.merge(planting_df, on = 'Plant Date', how='inner').reset_index(drop=True)
df.drop(columns=['Column2', 'Column1', 'Column3', 'Column4'], inplace = True)
#%%
weather_df.drop(columns=['Wind Speed [max]', 'Battery Voltage [last]', 
'Air Temperature [max]', 'Air Temperature [min]', 'Dew Point [min]'], inplace= True)
#%%
weather_df['dates'] = weather_df['Unnamed: 0'].astype(object)
weather_df['dates'] = pd.to_datetime(weather_df['dates'])
# weather_df.info()
#%%
weather_df['day'] = weather_df['dates'].dt.day
weather_df['month'] = weather_df['dates'].dt.month
weather_df['year'] = weather_df['dates'].dt.year
# %%
weather_df_2014 = weather_df[weather_df['year'] == 2014]
weather_df_2015 = weather_df[weather_df['year'] == 2015]
weather_df_2016 = weather_df[weather_df['year'] == 2016]
weather_df_2017 = weather_df[weather_df['year'] == 2017]
weather_df_2018 = weather_df[weather_df['year'] == 2018]
weather_df_2019 = weather_df[weather_df['year'] == 2019]
# %%
weather_df_2014.name = 'wet_2014'
weather_df_2015.name = 'wet_2015'
weather_df_2016.name = 'wet_2016'
weather_df_2017.name = 'wet_2017'
weather_df_2018.name = 'wet_2018'
weather_df_2019.name = 'wet_2019'

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
    i.drop(columns=['year', 'Unnamed: 0', 'dates', 'ET0 [result]'], inplace=True)
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
# %%
# %%
df_og = df
#%%
# restart point
df = df_og
# %%
df['plant_dates'] = pd.to_datetime(df['Plant Date'])
df['plant_dates_day'] = df['plant_dates'].dt.day
df['plant_dates_month'] = df['plant_dates'].dt.month
df['plant_dates_year'] = df['plant_dates'].dt.year
#%%
df['check_date'] = pd.to_datetime(df['Check Date'])
df['check_date_day'] = df['check_date'].dt.day
df['check_date_month'] = df['check_date'].dt.month
df['check_date_year'] = df['check_date'].dt.year
#%%
df.columns
# %%
df_y = df[['Head Weight (g)', 'Polar Diameter (mm)', 'Radial Diameter (mm)']].copy()
# 'Head Weight (g)', 'Polar Diameter (mm)', 'Radial Diameter (mm)'
df.drop(columns=['Diameter Ratio', 'Density (kg/L)', 'Head Weight (g)', 'Polar Diameter (mm)', 'Radial Diameter (mm)'], inplace=True)
#%%
# %%
l1 = df['Flight Date_x'].tolist()
l2 = df['Flight Date_y'].tolist()

l3=list()
for cnt, i in enumerate(l1):
    if i is pd.NaT:
        l3.append(l2[cnt])
    else:
        l3.append(i)
df['flight_mod'] = l3
df.to_csv('../dataset/tmp4.csv')
#%%
# %%
df = pd.read_csv('../dataset/tmp4.csv')
#%%
df['flight_mod'] = pd.to_datetime(df['flight_mod'])
df['plant_dates'] = pd.to_datetime(df['plant_dates'])

df['day_to_check'] = (df['flight_mod'] - df['plant_dates']).dt.days
df.info()
#%%
df.to_csv('../dataset/tmp5.csv')
#%%
# df.info()
df['flight_mod'] = pd.to_datetime(df['flight_mod'])
df['flight_mod_day'] = df['flight_mod'].dt.day
df['flight_mod_month'] = df['flight_mod'].dt.month
df['flight_mod_year'] = df['flight_mod'].dt.year
df = df[df.Remove != 'r']
df.drop(columns=['Diameter Ratio', 'Density (kg/L)', 'Head Weight (g)', 'Polar Diameter (mm)', 'Radial Diameter (mm)','Unnamed: 0', 'index', 'Flight Date_x', 'Flight Date_y', 'flight_mod', 'Plant Date', 'Check Date', 'plant_dates', 'check_date', 'Remove'], inplace= True)
df.info()
# df.to_csv('../dataset/tmp4_1.csv')
#%%
df.reset_index(drop=True)

# %%
for idx, row in df.iterrows():
    if idx >10 :
        break
    print(idx)
    df.loc[idx]['tmp'] \
        = datetime.datetime(
int(df.loc[idx]['flight_mod_year']),
int(df.loc[idx]['flight_mod_month']),
int(df.loc[idx]['flight_mod_day'])
            )
# %%
print(int(df.iloc[idx]['flight_mod_year']))
print(int(df.iloc[idx]['flight_mod_month']))
print(int(df.iloc[idx]['flight_mod_day']))
df.info()
# %%
df.head()

# %%
