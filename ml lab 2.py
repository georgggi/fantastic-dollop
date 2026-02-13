import pandas as pd

df = pd.read_csv('titanic.csv')
'''total_salary = df['Monthly_Salary'].sum()
average_salary = df['Monthly_Salary'].mean()
print('Total monthly salary: ', total_salary)
print('Average monthly salary: ', average_salary)'''
print(df.head())
print(df.info())
print(df.describe())
print(df.hist)
print(df.boxplot)
print(df.corr)
