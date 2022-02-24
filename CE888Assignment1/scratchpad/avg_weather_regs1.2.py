#%%
from datetime import datetime
from operator import le
from os import stat
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
#%%
weather_df = pd.read_excel('../dataset/Data.xlsx', sheet_name='weather')
# %%
# merge data from plants sheet
df = plants_df.merge(flights_df, on='Batch Number', how='inner').reset_index()
# df.to_csv('../dataset/plants_flights1.csv')
#%%
df.info()
#%%
# change column name and convert to datetime format from numpy datetime64 format
planting_df.rename(columns={"Plant_Date": "Plant Date"}, inplace=True)
# convert to datetime format from numpy datetime64 format
planting_df['Plant Date'] = pd.to_datetime(planting_df['Plant Date'], errors = 'coerce')
df['Plant Date'] = pd.to_datetime(df['Plant Date'], errors = 'coerce')
#%%
# get data from planting sheet and merge with plants sheet using 'Plant Date' as key
df = df.merge(planting_df, on = 'Plant Date', how='inner').reset_index(drop=True)
# %%
# %%
df_og = df
# restart point
df = df_og
# %%
# split date to day, month and year
df['plant_dates'] = pd.to_datetime(df['Plant Date'])
df['plant_dates_day'] = df['plant_dates'].dt.day
df['plant_dates_month'] = df['plant_dates'].dt.month
df['plant_dates_year'] = df['plant_dates'].dt.year
#%%
# split date to day, month and year
df['check_date'] = pd.to_datetime(df['Check Date'])
df['check_date_day'] = df['check_date'].dt.day
df['check_date_month'] = df['check_date'].dt.month
df['check_date_year'] = df['check_date'].dt.year

# %%
# snippet to fill flight dates
l1 = df['Flight Date_x'].tolist()
l2 = df['Flight Date_y'].tolist()
l3=list()
for cnt, i in enumerate(l1):
    if i is pd.NaT:
        l3.append(l2[cnt])
    else:
        l3.append(i)
df['flight_mod'] = l3
#%%
# create new feature for no of days from plant date to flight date
df['flight_mod'] = pd.to_datetime(df['flight_mod'])
df['plant_dates'] = pd.to_datetime(df['plant_dates'])
df['day_to_check'] = (df['flight_mod'] - df['plant_dates']).dt.days
#%%

#%%
# df.to_csv('../dataset/tmp5.csv')
#%%
# df.info()
df['flight_mod_day'] = df['flight_mod'].dt.day
df['flight_mod_month'] = df['flight_mod'].dt.month
df['flight_mod_year'] = df['flight_mod'].dt.year
df = df[df.Remove != 'r']
#%%
df.to_csv('../dataset/undropped1.csv')
#%%
df = pd.read_csv('../dataset/undropped1.csv')
df.info()
#%%
df.drop(columns=['Remove', 'Flight Date_y' ,'Flight Date_x', 'Column1', 'Column2', 'Column3', 'Column4'], inplace=True)
df_corr_check = df
df_corr_check.info()
#%%
pd.set_option('display.expand_frame_repr', False)

xtrain_null = df.loc[df['Radial Diameter (mm)'].isnull(), df.columns]
xtrain_notnull = df.loc[df['Radial Diameter (mm)'].notnull(), df.columns]
xtrain_notnull.head()

# xtrain.info()
#%%
# Create correlation matrix
corr_matrix = xtrain_notnull.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
corr_matrix
#%%
to_drop
#%%
# df_corr_check.corr()
df_corr_check.head()
#%%
# impute missing values of 'Head Weight (g)', 'Polar Diameter (mm)', 'Radial Diameter (mm)'

# %%
# get targets for multi label regression
df_y = df[['Head Weight (g)', 'Polar Diameter (mm)', 'Radial Diameter (mm)']].copy()
#%%
df.drop(columns=['Column2', 'Column1', 'Column3', 'Column4', 'Diameter Ratio', 'Density (kg/L)', 'Head Weight (g)', 'Polar Diameter (mm)', 'Radial Diameter (mm)','Unnamed: 0', 'index', 'Flight Date_x', 'Flight Date_y', 'flight_mod', 'Plant Date', 'Check Date', 'plant_dates', 'check_date', 'Remove'], inplace= True)
# df.to_csv('../dataset/tmp4_1.csv')
#%%
df.reset_index(drop=True)
df.info()


# %%



































#%%
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

#%%
df.dropna(axis=0, subset = ['Crop'], inplace=True)
df.info()
#%%
start_date = list(zip(df['plant_dates_month'],df['plant_dates_day']))
end_date = list(zip(df['check_date_month'],df['check_date_day']))
day_to_check_lst = df.day_to_check
print(len(start_date), len(end_date), len(day_to_check_lst))

start_wet = []
end_wet = []
mid_wet = []
#%%
a = (datetime(2000, int(start_date[0][0]), int(start_date[0][1])))
b = (datetime(2000, int(end_date[0][0]), int(end_date[0][1])))
print(a)
print(b)
# print(day_to_check_lst[0])
print(a + (b - a)/2)
#%%
# for i in range(len(start_date)):
a = (datetime(2000, int(start_date[0][0]), int(start_date[0][1])))
b = (datetime(2000, int(end_date[0][0]), int(end_date[0][1])))
mid_date = (a + (b - a)/2)
mid_day = mid_date.day
mid_month = mid_date.month
print(mid_day, mid_month)
#%%

st_wet = model_list[0].predict([start_date[0]])
ed_wet = model_list[0].predict([end_date[0]])
md_wet = model_list[0].predict([[mid_month, mid_day]])

x = np.mean(np.array([st_wet[0],
md_wet[0],
ed_wet[0]]), axis = 0)
# print(st_wet[0])
# print(md_wet[0])
# print(ed_wet[0])
x
#%%
x_st = []
x_ed = []
x_md = []
for i in range(len(start_date)):
    if i % 1000 == 0:
        print(f'processed {i} rows')
    st_wet = []
    ed_wet = []
    md_wet = []
    a = (datetime(2000, int(start_date[i][0]), int(start_date[i][1])))
    b = (datetime(2000, int(end_date[i][0]), int(end_date[i][1])))
    mid_date = (a + (b - a)/2)
    mid_day = mid_date.day
    mid_month = mid_date.month
    # print(a,'\n', b, '\n', mid_date)
    # print(mid_day, mid_month)
    for j in range(len(model_list)):
        st_wet.append(model_list[j].predict([start_date[i]]))
        # print(model_list[j].predict([start_date[i]]))
        ed_wet.append(model_list[j].predict([end_date[i]]))
        md_wet.append(model_list[j].predict([[mid_month, mid_day]]))
    x_st.append(np.mean(np.array(st_wet), axis = 0))
    x_ed.append(np.mean(np.array(ed_wet), axis = 0))
    x_md.append(np.mean(np.array(md_wet), axis = 0))
    # print(x_st)
    # print('*******')
# print(st_wet[0])
# print(md_wet[0])
# print(ed_wet[0])
# x
#%%
df['start_wt'] = x_st

# %%
