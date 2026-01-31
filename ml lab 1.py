import pandas as pd

df = pd.read_csv('employee_salary_dataset.csv')
total_salary = df['Monthly_Salary'].sum()
average_salary = df['Monthly_Salary'].mean()
print('Total monthly salary: ', total_salary)
print('Average monthly salary: ', average_salary)
