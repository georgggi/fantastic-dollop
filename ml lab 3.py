import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Импорт моделей машинного обучения
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Импорт функции разделения данных
from sklearn.model_selection import train_test_split

# Импорт метрик оценки модели
from sklearn.metrics import mean_squared_error as mse, r2_score as r2

# Настройка отображения чисел (2 знака после запятой)
pd.set_option('display.float_format', '{:.2f}'.format)

# Отключение научной записи чисел
np.set_printoptions(suppress=True)

# Загрузка датасета и удаление столбца region
df = pd.read_csv('insurance.csv').drop('region', axis=1)

# Вывод количества признаков и строк
print('Features: ', df.shape[1], '\nTarget: ', df.shape[0])

# Расчет количества тестовых и обучающих данных
print('\nNum of testing datas: ', round(df.shape[0] * 0.2), '\nNum of learning datas: ', round(df.shape[0] * 0.8))

# Вывод датасета
print(df)

# Преобразование категориальных данных в числовые
df = df.assign(smoker=df['smoker'].map({'no': 0, 'yes': 1}), sex=df['sex'].map({'female': 0, 'male': 1}))

# Расчет корреляции между числовыми признаками и целевой переменной charges
ts = df.select_dtypes([int, float]).corr()[['charges']].drop('charges').T

# Список признаков для анализа
listed = ['sex', 'age', 'bmi', 'children', 'smoker']

# Удаление признаков со слабой корреляцией (< 0.1)
for i in listed.copy():

    # коэффициент корреляции
    s = round(ts.loc['charges', i], 2)
    if abs(s) < 0.1:
        listed.remove(i)
        df = df.drop(i, axis=1)
        ts = ts.drop(i, axis=1)
        continue

print(df)

# Вывод признаков, которые больше всего влияют на charges
print("\nMore dependences in:", ", ".join(listed))

# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(
    df.drop('charges', axis=1),  # признаки
    df['charges'],                             # целевая переменная
    test_size=0.2,                             # 20% тест
    random_state=42)

# Создание моделей: линейная регрессия, дерево решений
lr, dt = LinearRegression(), DecisionTreeRegressor(max_depth=4, random_state=42)

# Обучение моделей
lr.fit(x_train, y_train), dt.fit(x_train, y_train)

# Предсказания моделей
pred_lr, pred_dt = np.maximum(lr.predict(x_test), 0), np.maximum(dt.predict(x_test), 0)

# Расчет ошибок (MSE)
mse_lr, mse_dt = mse(y_test, pred_lr), mse(y_test, pred_dt)

# Расчет качества модели (R²)
r2_lr, r2_dt = r2(y_test, pred_lr), r2(y_test, pred_dt)

# Вывод метрик моделей
print("\nModel metrics:")
print(f"Linear Regression -> MSE: {mse_lr:.2f}, R2: {r2_lr:.3f}")
print(f"Decision Tree     -> MSE: {mse_dt:.2f}, R2: {r2_dt:.3f}")

# ---------- Визуализация ----------

# Тепловая карта корреляции
plt.figure()
sns.heatmap(ts, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation with charges")
plt.show()

# Сравнение моделей по MSE
plt.figure()
plt.bar(['Linear Regression', 'Decision Tree'], [mse_lr, mse_dt])
plt.title("MSE comparison")
plt.ylabel("MSE")
plt.show()

# Сравнение моделей по R2
plt.figure()
plt.bar(['Linear Regression', 'Decision Tree'], [r2_lr, r2_dt])
plt.title("R2 comparison")
plt.ylabel("R2")
plt.show()

# График: реальные значения vs предсказанные
plt.figure()
plt.scatter(y_test, pred_lr, alpha=0.6, label="Linear Regression")
plt.scatter(y_test, pred_dt, alpha=0.6, label="Decision Tree")

# Диагональная линия идеального предсказания
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")
plt.xlabel("Real charges")
plt.ylabel("Predicted charges")
plt.title("Real vs Predicted")
plt.legend()
plt.grid(True)
plt.show()
