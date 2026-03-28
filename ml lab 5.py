import pandas as pd; df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv') # Загрузка датасета

# Импорты моделей и инструментов
from sklearn.preprocessing import LabelEncoder as le
from sklearn.ensemble import VotingClassifier as v, RandomForestClassifier as rf
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.model_selection import train_test_split as tts, cross_validate as cv
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# Преобразование категориального признака транспорта (one-hot encoding)
op = pd.get_dummies(df['MTRANS'])       # создаём бинарные признаки для каждого типа транспорта
df = df.drop('MTRANS', axis=1)    # удаляем исходный столбец
df = pd.concat([df, op], axis=1)   # добавляем новые one-hot признаки

# Кодирование категориальных признаков
for col in df.select_dtypes([str, bool]).columns.drop(['CAEC', 'CALC', 'NObeyesdad']):
    df[col] = le().fit_transform(df[col])
for col in ['CAEC', 'CALC']:
    df[col] = df[col].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})

# Кодирование целевой переменной
df['NObeyesdad'] = df['NObeyesdad'].map({'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2,
                                         'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6})

# Анализ структуры данных
classes = []
for i in df.dtypes:
    if i not in classes: classes.append(i)
print(f'Number of features {df.shape[1]}, number of classes {len(classes)}, number of selection for train 0.8')

# Вычисляем корреляцию всех признаков с целевой переменной
df_corr = df.corr()[['NObeyesdad']].T

# Удаляем признаки с очень слабой линейной зависимостью
for feat in df.columns:
    if feat != 'NObeyesdad':

        # коэффициент корреляции
        s = round(df_corr.loc['NObeyesdad', feat], 2)

        # если корреляция слабая — удаляем признак
        if abs(s) < 0.1:
            df, df_corr = df.drop(feat, axis=1), df_corr.drop(feat, axis=1)

# Разделение на признаки и целевую переменную
x, y = df.drop('NObeyesdad', axis=1), df['NObeyesdad']

# Train/Test split (80/20)
# Lewis Hamilton - 44 Ferrari
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=44)

# Инициализация моделей
clDT, clKNN, clRF = dt(random_state=44), knn(n_neighbors=5), rf(random_state=44)

# Обучение моделей
clDT.fit(x_train, y_train), clKNN.fit(x_train, y_train), clRF.fit(x_train, y_train)

# Оценка accuracy на тестовой выборке
DT_acc, KNN_acc, RF_acc = clDT.score(x_test, y_test), clKNN.score(x_test, y_test), clRF.score(x_test, y_test)
print(f'Accuracy for models: DesicionTree - {DT_acc * 100:.1f}%; KNN - {KNN_acc * 100:.1f}%, RandomForest - {RF_acc * 100:.1f}%')

# Ансамбль моделей (Voting Classifier)
clV = v(estimators=[('DT', clDT), ('KNN', clKNN), ('RF', clRF)], voting='hard')

clV.fit(x_train, y_train)

# Предсказания ансамбля
VC_acc, y_pred = clV.score(x_test, y_test), clV.predict(x_test)
print(f'Accuracy VC: {VC_acc * 100:.1f}%. \n'
      f'Δ(VC, DT) = {(VC_acc - DT_acc) * 100:.1f}%; Δ(VC, KNN) = {(VC_acc - KNN_acc) * 100:.1f}%; Δ(VC, RF) = {(VC_acc - RF_acc) * 100:.1f}%')

# Кросс-валидация (5-fold)
metrics_results = []
for name, model in [('DesicionTree', clDT), ('KNN', clKNN), ('RandomForest', clRF), ('Voting', clV)]:
    cv_results = cv(model, x, y, cv=5, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])

    # Сохраняем усреднённые результаты
    metrics_results.append({
        'Model': name,
        'Accuracy': cv_results['test_accuracy'].mean(),
        'Precision': cv_results['test_precision_macro'].mean(),
        'Recall': cv_results['test_recall_macro'].mean(),
        'F1-Score': cv_results['test_f1_macro'].mean()
    })

# Таблица сравнения моделей
df_metrics = pd.DataFrame(metrics_results)
print('Model Comparison Table')
print(df_metrics.to_string(index=False))

# Матрицы ошибок
for name, model in [('DesicionTree', clDT), ('KNN', clKNN), ('RandomForest', clRF), ('Voting', clV)]:
    plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap='Greens')
    plt.title(f'Confusion matrix: {name}')
    plt.show()

# Важность признаков
importances = pd.Series(clRF.feature_importances_, index=x.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=importances.index, palette='Blues')
plt.title('Feature Importance (Важность признаков) - Random Forest')
plt.xlabel('Относительная важность')
plt.show()