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

    return data_mean, lower, upper

def bootstrap_mean_ci(x, sample_size, n_bootstraps, cf):
    b_array = np.random.choice(x, (n_bootstraps, sample_size))
    data_mean = np.mean(b_array, axis=1)
    lower, upper = np.percentile(data_mean, [(100-cf)/2, cf+((100-cf)/2)])
    data_mean = np.mean(b_array)

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
df_boot.iloc[10]

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
#%%
df = pd.read_csv('https://raw.githubusercontent.com/albanda/CE888/master/lab2%20-%20bootstrap/vehicles.csv')
data_old = df.values.T[0]
data_new = df.values.T[1]

# %%
boots = []
for i in range(100, 50000, 1000):
    boot = bootstrap_mean_ci(data_new, data_new.shape[0], i, 80)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

sns_plot.axes[0, 0].set_ylim(0,)
sns_plot.axes[0, 0].set_xlim(0, 50000)

# %%
