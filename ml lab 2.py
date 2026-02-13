import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
df = pd.read_csv('titanic.csv')

print(df.select_dtypes(int).describe())
'''print(df['Age'].hist(bins=50, figsize=(10, 5)))
print(df.boxplot(column='Age', figsize=(10, 4)))
print(df.plot.scatter(x='Age', y='Survived'))
print(df.select_dtypes(int).corr())
sns.histplot(df['Age'], bins=50)'''
sns.scatterplot(x='Age', y='Survived', data=df)
plt.show()