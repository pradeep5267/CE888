#%%
from turtle import up
import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np

#%%
df = pd.read_csv('https://raw.githubusercontent.com/albanda/CE888/master/lab2%20-%20bootstrap/customers.csv')
data = df.values.T[1]

#%%
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
    # print(data_mean, lower, upper)


    return data_mean, lower, upper

def bootstrap_mean_ci(x, sample_size, n_bootstraps, cf):
    b_array = np.random.choice(x, (n_bootstraps, sample_size))
    data_mean = np.mean(b_array, axis=1)
    lower, upper = np.percentile(data_mean, [(100-cf)/2, cf+((100-cf)/2)])
    data_mean = np.mean(b_array)
    # print(data_mean, lower, upper)

    return data_mean, lower, upper


# %%
# Call your bootstrap function and plot the results

boots = []
for i in range(100, 50000, 1000):
    # print(f'inside ------- {data.shape[0]}, {i}')
    boot = bootstrap_mean(data, data.shape[0], i)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])
#%%
# print(len(df_boot.columns[0]), len(df_boot.columns[1]))
#%%
df_boot = pd.DataFrame(boots, columns=['Bootstrap Iterations', 'Mean', "Value"])
# df_boot.iloc[10]

#%%
sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")
sns_plot.axes[0, 0].set_ylim(0,)
sns_plot.axes[0, 0].set_xlim(0, 50000)


#%%
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
# %%

# #%%
# df['New Fleet'].plot.kde()
# #%%
# df.describe()
#%%
df = pd.read_csv('https://raw.githubusercontent.com/albanda/CE888/master/lab2%20-%20bootstrap/vehicles.csv')
data_old = df["Current fleet"].T
data_new = df.dropna(axis=0)
data_new = data_new['New Fleet'].T
sns.scatterplot(y = np.linspace(0,50,len(data_old)), x = data_old)
sns.scatterplot(y = np.linspace(0,50,len(data_new)), x = data_new)

#%%
print(data_new.shape, data_old.shape)
#%%
def bootstrap_mean_ci(x, sample_size, n_bootstraps, cf):
    b_array = np.random.choice(x, (n_bootstraps, sample_size))
    data_mean = np.mean(b_array, axis=1)
    lower, upper = np.percentile(data_mean, [(100-cf)/2, cf+((100-cf)/2)])
    data_mean = np.mean(b_array)
    # print(data_mean, lower, upper)

    return data_mean, lower, upper
# %%
boots = []
for i in range(100, 50000, 1000):
    boot = bootstrap_mean_ci(data_new, data_new.shape[0], i, 95)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_boot_new = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])

boots = []
for i in range(100, 50000, 1000):
    boot = bootstrap_mean_ci(data_old, data_old.shape[0], i, 95)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_boot_old = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
#%%
print(f'shapes ==== {df_boot_old.shape}, {df_boot_new.shape}')
print(f'shapes ==== {df_boot_old.Mean.mean()}, {df_boot_new.Mean.mean()}')
t_obs = df_boot_new.Mean.mean() - df_boot_old.Mean.mean()
print(f't_obs ==== {t_obs}')
#%%
sns_plot = sns.lmplot(x = 'Boostrap Iterations', y = 'Mean', data=df_boot_new, fit_reg=False, hue="Value")

sns_plot.axes[0, 0].set_ylim(0,)
sns_plot.axes[0, 0].set_xlim(0, 50000)


sns_plot = sns.lmplot(x = 'Boostrap Iterations', y = 'Mean', data=df_boot_old, fit_reg=False, hue="Value")

sns_plot.axes[0, 0].set_ylim(0,)
sns_plot.axes[0, 0].set_xlim(0, 50000)
#%%
df_boot_new.describe()
#%%
t_obs/50000
# df_obs = pd.DataFrame(df_boot_new['Mean'] - df_boot_old['Mean'], columns=['t obs'])
#%%
df_obs = pd.DataFrame(data=df_obs, columns=['Mean'])
df_obs.head()
#%%
# l = df_obs['Mean'] > 20
print(type(l), l.value_counts(), len(l), df_obs.shape)
#%%
# (df_obs['Mean'] > 20)
df_obs[df_obs > 10.0].count()

# %%



def permute_mean(x1, x2):
    np.random.shuffle(x1)
    np.random.shuffle(x2)
    print(type(x1), type(x2), len(x1), len(x2))
    return np.abs(np.mean(x1) - np.mean(x2))
    # a_mean = np.mean(x1)
    # b_mean = np.mean(x2)
    # print(a_mean, b_mean)
#%%
# Create your own function for a permutation test here (you will need it for the lab quiz!):
def permut_test(sample1, sample2, n_permutations):
    """
    sample1: 1D array
    sample2: 1D array (note that the size of the two arrays can be different)
    n_permutations: number of permutations to calculate the p-value
    """

    
    ''' keep sample1 as the bigger array'''
    # print(type(sample1), type(sample2), len(sample1), len(sample2))
    for i in range(n_permutations):
        if len(sample1) < len(sample2):
            sample1, sample2 = sample2, sample1
        s1 = np.random.choice(sample1, size=len(sample2))
        s2 = np.random.choice(sample2, size=len(sample2))
        # print(type(s1), type(s2), len(s1), len(s2))
        s = np.concatenate([s1, s2])
        # print(len(s)/2)
        s_old = s[:int(len(s)/2)]
        s_new = s[int(len(s)/2):]
        # print(type(s_old), type(s_new), len(s_old), len(s_new))

        t_perm = permute_mean(s_old, s_new)
        print(t_perm)
    # return pvalue
# %%
permut_test(df_boot_new.Mean.values, df_boot_old.Mean.values, 3)
# %%
