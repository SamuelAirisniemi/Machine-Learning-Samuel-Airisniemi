import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset("mpg")
df_describe = df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']].describe()
#print(df_describe)

#sns.pairplot(df[["mpg", "weight", "horsepower", "displacement", "cylinders"]])

fig, axes = plt.subplots(2, 3)
axes = axes.flatten()

sns.scatterplot(x = "horsepower", y = "weight", data = df, ax = axes[0])
sns.scatterplot(x = "acceleration", y = "weight", data = df, ax = axes[1])
sns.scatterplot(x = "cylinders", y = "weight", data = df, ax = axes[2])
sns.scatterplot(x = "mpg", y = "weight", data = df, ax = axes[3])
sns.scatterplot(x = "mpg", y = "horsepower", data = df, ax = axes[4])
sns.scatterplot(x = "mpg", y = "displacement", data = df, ax = axes[5])

plt.tight_layout()
plt.show()
