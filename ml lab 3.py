import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression as lr
import seaborn as sns

df = pd.read_csv('insurance.csv')
print('Features: ', df.shape[1], '\nTarget: ', df.shape[0])
testing = sorted(np.random.choice(np.arange(0, df.shape[0]), size=round(0.2 * df.shape[0]), replace=False).tolist())
learning = [i for i in range(df.shape[0]) if i not in testing]
print('\nNum of testing datas: ', len(testing))
print('Num of learning datas: ', len(learning))
'''
#print('Linear Regression')
#model = lr()
#model.fit(df['charges'], df['age'])
#y_pred = model.predict(df['charges'])
#plt.plot(df['charges'], y_pred, color='red')
print(df.select_dtypes([int, float]))
sns.histplot(df['age'], bins=50)
sns.heatmap(df.select_dtypes([int, float]))
plt.show()
sns.scatterplot(x='age', y='charges', data=df)

print(df.shape[0] - df[df['bmi'] < 30].shape[0])
print(df[['sex', 'charges']].groupby('sex').agg(['min', 'max', 'median', 'mean', 'sum']).plot(subplots=True))
plt.plot()
plt.show()
print(df.groupby('sex')['sex'].count().plot())

plt.show()'''

print(df)
#print(df.groupby('smoker')['charges'].agg(['sum', 'mean', 'count', 'median']).iloc[:, :].plot.line(marker='v', subplots=True))
op = df.select_dtypes([int, float]).iloc[:, :]
nd, nr, iz, ozh = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
nd, nr, iz, ozh = (df[df['bmi'] < 18.5], df[(df['bmi'] >= 18.5) & (df['bmi'] <= 24.9)],
                   df[(df['bmi'] >= 25) & (df['bmi'] <= 29.9)], df[df['bmi'] >= 30])

pd.set_option('display.float_format', '{:.2f}'.format)
listed = {'Недостаток веса': nd, 'Норма': nr, 'Избыток веса': iz, 'Ожирение': ozh}
ch = pd.DataFrame(index=[['sum', 'mean', 'age_mean']])
for i in listed:
    print('\n', i)
    #print(listed[i])
    #print(listed[i].shape)
    print(listed[i].groupby('region')['charges'].agg(['sum', 'mean']))
    print(listed[i].pivot(columns='sex', values='charges').plot(subplots=True))
    print(listed[i].pivot_table(columns='sex', index='age', values='charges', aggfunc='mean'))
#    print(listed[i].groupby(['smoker', 'sex'])['charges'].agg(['mean', 'sum', 'count']).plot.area(subplots=True))
    ch[i] = [listed[i]['charges'].sum(), listed[i]['charges'].mean(), listed[i]['age'].mean()]
    #sns.heatmap(listed[i].select_dtypes([int, float]))

#print(ch.T.plot(subplots=True))
#print(df.groupby(['smoker', 'sex', 'region']).agg({'age' :['mean'], 'bmi':['mean', 'median', 'min'],
#                                                   'charges': ['sum', 'mean']}).plot.line(marker='v', subplots=True, ))
#print(df.groupby(['smoker', 'sex', 'region']).agg({'age' :['mean'], 'bmi':['mean', 'median', 'min'],
#                                                   'charges': ['sum', 'mean']}))
#print(df.groupby(['bmi']).agg({'charges': ['sum', 'mean', 'count']}).plot(subplots=True, marker='p'))
#print(df.iloc[:, :5].groupby('smoker').agg(['sum', 'mean', 'count']))
#sns.heatmap(df.select_dtypes([int, float]))
plt.show()
