#%%
from turtle import up
import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np


df = pd.read_csv('https://raw.githubusercontent.com/albanda/CE888/master/lab2%20-%20bootstrap/customers.csv')
data = df.values.T[1]


# Checking the notes from the lecture, create here your own bootstrap function:
# 1. Sample from the input array x to create an array of samples of shape (n_bootstraps, sample_size)
# Hint: Check the function random.choice() on Numpy
# 2. Calculate and save the mean of the array (this is "data_mean" that is returned by the function)
# 3. Calculate the mean from each bootstrap (i.e., row) and store it.
# (This should be an array of n_bootstraps values)
# 4. Calculate the lower and upper bounds for a 95% CI (hint: check the percentile function on Numpy)
# 5. Return data_mean, and the lower and upper bounds of your interval
def bootstrap_mean(x, sample_size, n_bootstraps):
	# <---INSERT YOUR CODE HERE--->
    b_array = np.random.choice(x, (n_bootstraps, sample_size))
    data_mean = np.mean(b_array, axis=1)
    lower, upper = np.percentile(data_mean, [(100-95)/2, 95+((100-95)/2)])
    data_mean = np.mean(b_array)
    return data_mean, lower, upper

def bootstrap_mean_ci(x, sample_size, n_bootstraps, cf):
    b_array = np.random.choice(x, (n_bootstraps, sample_size))
    data_mean = np.mean(b_array, axis=1)
    lower, upper = np.percentile(data_mean, [(100-cf)/2, cf+((100-cf)/2)])
    data_mean = np.mean(b_array)
    return data_mean, lower, upper

#%%
def bootstrap_std(x, sample_size, n_bootstraps):
	# <---INSERT YOUR CODE HERE--->
    b_array = np.random.choice(x, (n_bootstraps, sample_size))
    data_mean = np.std(b_array, axis=1)
    lower, upper = np.percentile(data_mean, [(100-95)/2, 95+((100-95)/2)])
    data_mean = np.std(b_array)
    return data_mean, lower, upper

def bootstrap_std_ci(x, sample_size, n_bootstraps, cf):
    b_array = np.random.choice(x, (n_bootstraps, sample_size))
    data_mean = np.std(b_array, axis=1)
    lower, upper = np.percentile(data_mean, [(100-cf)/2, cf+((100-cf)/2)])
    data_mean = np.std(b_array)
    return data_mean, lower, upper

# %%
# Call your bootstrap function and plot the results
boots = []
for i in range(100, 50000, 1000):
    boot = bootstrap_mean(data, data.shape[0], i)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])
df_boot = pd.DataFrame(boots, columns=['Bootstrap Iterations', 'Mean', "Value"])
sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")
sns_plot.axes[0, 0].set_ylim(0,)
sns_plot.axes[0, 0].set_xlim(0, 50000)

boots = []
for i in range(100, 50000, 1000):
    boot = bootstrap_mean_ci(data, data.shape[0], i, 80)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

sns_plot.axes[0, 0].set_ylim(0,)
sns_plot.axes[0, 0].set_xlim(0, 50000)

sns_plot.savefig("bootstrap_confidence_80.pdf", bbox_inches='tight')
#%%
df = pd.read_csv('https://raw.githubusercontent.com/albanda/CE888/master/lab2%20-%20bootstrap/vehicles.csv')
data_old = df["Current fleet"].T
data_new = df.dropna(axis=0)
data_new = data_new['New Fleet'].T
# sns.scatterplot(y = np.linspace(0,50,len(data_old)), x = data_old)
# sns.scatterplot(y = np.linspace(0,50,len(data_new)), x = data_new)
print(f' {data_new.shape}, {data_old.shape}')
# %%
boots = []
for i in range(1,10000):
    boot = bootstrap_mean_ci(data_new, data_new.shape[0], i, 95)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_mean_95_new = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

boots = []
for i in range(1,10000):
    boot = bootstrap_mean_ci(data_old, data_old.shape[0], i, 85)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_mean_85_old = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

# print(f'shapes ==== {df_mean_old.shape}, {df_mean_new.shape}')
# print(f'shapes ==== {df_mean_old.Mean.mean()}, {df_mean_new.Mean.mean()}')
# t_obs = df_mean_new.Mean.mean() - df_mean_old.Mean.mean()
# print(f't_obs ==== {t_obs}')
#%%
df_mean_95_new.describe()
#%%
df_mean_85_old.describe()

#%%
boots = []
for i in range(1,10000):
    boot = bootstrap_std_ci(data_new, data_new.shape[0], i, 95)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_std_95_new = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

boots = []
for i in range(1,10000):
    boot = bootstrap_std_ci(data_old, data_old.shape[0], i, 73)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_std_73_old = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

# print(f'shapes ==== {df_mean_old.shape}, {df_mean_new.shape}')
# print(f'shapes ==== {df_mean_old.Mean.mean()}, {df_mean_new.Mean.mean()}')
# t_obs = df_mean_new.Mean.mean() - df_mean_old.Mean.mean()
# print(f't_obs ==== {t_obs}')
#%%
df_std_95_new.describe()
#%%
df_std_73_old.describe()


#%%

df_mean_95_new.head()
#%%
# df_meann = df_mean_95_new.groupby(by = "Value").min()
# df_meann.head()


df_mean_95_new.groupby(by = "Value").max()










#%%
boots = []
for i in range(1,10000):
    boot = bootstrap_mean_ci(data_old, data_old.shape[0], i, 73)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_boot_old = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

# print(f'shapes ==== {df_boot_old.shape}, {df_boot_new.shape}')
# print(f'shapes ==== {df_boot_old.Mean.mean()}, {df_boot_new.Mean.mean()}')
# t_obs = df_boot_new.Mean.mean() - df_boot_old.Mean.mean()
# print(f't_obs ==== {t_obs}')

df_boot_old.describe()
#%%


# %%
boots = []
for i in range(1,10000):
    boot = bootstrap_mean_ci(data_new, data_new.shape[0], i, 95)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_boot_new = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

boots = []
for i in range(1,10000):
    boot = bootstrap_mean_ci(data_old, data_old.shape[0], i, 95)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_boot_old = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

print(f'shapes ==== {df_boot_old.shape}, {df_boot_new.shape}')
print(f'shapes ==== {df_boot_old.Mean.mean()}, {df_boot_new.Mean.mean()}')
t_obs = df_boot_new.Mean.mean() - df_boot_old.Mean.mean()
print(f't_obs ==== {t_obs}')
#%%
df_boot_new.describe()
#%%
df_boot_old.describe()










#%%
sns_plot = sns.lmplot(x = 'Boostrap Iterations', y = 'Mean', data=df_boot_new, fit_reg=False, hue="Value")
sns_plot.axes[0, 0].set_ylim(0,)
sns_plot.axes[0, 0].set_xlim(0, 50000)

sns_plot = sns.lmplot(x = 'Boostrap Iterations', y = 'Mean', data=df_boot_old, fit_reg=False, hue="Value")
sns_plot.axes[0, 0].set_ylim(0,)
sns_plot.axes[0, 0].set_xlim(0, 50000)
#%%

# print(df_boot_new.describe())
# df_obs = pd.DataFrame(df_boot_new['Mean'] - df_boot_old['Mean'], columns=['t obs'])

print(t_obs, t_obs/50)

def permute_mean(x1, x2):
    np.random.shuffle(x1)
    np.random.shuffle(x2)
    return np.abs(np.mean(x1) - np.mean(x2))
#%%
# Create your own function for a permutation test here (you will need it for the lab quiz!):
def permut_test(sample1, sample2, n_permutations, t_obs):
    """
    sample1: 1D array
    sample2: 1D array (note that the size of the two arrays can be different)
    n_permutations: number of permutations to calculate the p-value
    """
    t_perm_array = []

    ''' keep sample1 as the bigger array'''
    for i in range(n_permutations):
        if len(sample1) < len(sample2):
            sample1, sample2 = sample2, sample1
        s1 = np.random.choice(sample1, size=len(sample2))
        s2 = np.random.choice(sample2, size=len(sample2))
        s = np.concatenate([s1, s2])
        s_old = s[:int(len(s)/2)]
        s_new = s[int(len(s)/2):]
        t_perm = permute_mean(s_old, s_new)
        t_perm_array.append(t_perm)
    
    z = ((np.array(t_perm_array) > t_obs).sum())/n_permutations
    print(z)

# %%
permut_test(df_boot_new.Mean.values, df_boot_old.Mean.values, 30000, t_obs)
# %%
# df = pd.read_csv('https://raw.githubusercontent.com/albanda/CE888/master/lab2%20-%20bootstrap/vehicles.csv')
# data_old = df["Current fleet"].T
# data_new = df.dropna(axis=0)
# data_new = data_new['New Fleet'].T
# print(f' {data_new.shape}, {data_old.shape}')
#%%
# The variables below represent the percentages of democratic votes in Pennsylvania and Ohio (one value for each state).
dem_share_PA = [60.08, 40.64, 36.07, 41.21, 31.04, 43.78, 44.08, 46.85, 44.71, 46.15, 63.10, 52.20, 43.18, 40.24, 39.92, 47.87, 37.77, 40.11, 49.85, 48.61, 38.62, 54.25, 34.84, 47.75, 43.82, 55.97, 58.23, 42.97, 42.38, 36.11, 37.53, 42.65, 50.96, 47.43, 56.24, 45.60, 46.39, 35.22, 48.56, 32.97, 57.88, 36.05, 37.72, 50.36, 32.12, 41.55, 54.66, 57.81, 54.58, 32.88, 54.37, 40.45, 47.61, 60.49, 43.11, 27.32, 44.03, 33.56, 37.26, 54.64, 43.12, 25.34, 49.79, 83.56, 40.09, 60.81, 49.81]
dem_share_OH = [56.94, 50.46, 65.99, 45.88, 42.23, 45.26, 57.01, 53.61, 59.10, 61.48, 43.43, 44.69, 54.59, 48.36, 45.89, 48.62, 43.92, 38.23, 28.79, 63.57, 38.07, 40.18, 43.05, 41.56, 42.49, 36.06, 52.76, 46.07, 39.43, 39.26, 47.47, 27.92, 38.01, 45.45, 29.07, 28.94, 51.28, 50.10, 39.84, 36.43, 35.71, 31.47, 47.01, 40.10, 48.76, 31.56, 39.86, 45.31, 35.47, 51.38, 46.33, 48.73, 41.77, 41.32, 48.46, 53.14, 34.01, 54.74, 40.67, 38.96, 46.29, 38.25, 6.80, 31.75, 46.33, 44.90, 33.57, 38.10, 39.67, 40.47, 49.44, 37.62, 36.71, 46.73, 42.20, 53.16, 52.40, 58.36, 68.02, 38.53, 34.58, 69.64, 60.50, 53.53, 36.54, 49.58, 41.97, 38.11]
print(f'OH = {len(dem_share_OH)},\n,PA = {len(dem_share_PA)}')

# %%
dem_share_OH, dem_share_PA = np.array(dem_share_OH), np.array(dem_share_PA)


boots = []
for i in range(1,20000):
    boot = bootstrap_mean_ci(dem_share_PA, dem_share_PA.shape[0], i, 95)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

PA_mean_95_new = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

boots = []
for i in range(1,20000):
    boot = bootstrap_mean_ci(dem_share_OH, dem_share_PA.shape[0], i, 95)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

OH_boot_old = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

print(f'shapes ==== {OH_boot_old.shape}, {PA_mean_95_new.shape}')
print(f'shapes ==== {OH_boot_old.Mean.mean()}, {PA_mean_95_new.Mean.mean()}')
t_obs = PA_mean_95_new.Mean.mean() - OH_boot_old.Mean.mean()
print(f't_obs ==== {t_obs}')
# %%
PA_mean_95_new.describe()
#%%
OH_boot_old.describe()
# %%
permut_test(PA_mean_95_new.Mean.values, OH_boot_old.Mean.values, 10000, t_obs)

