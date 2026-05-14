import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

df = pd.read_csv("restaurants.csv")

# Средний рейтинг по всему датасету
C = df["rating"].mean()

# Минимальный порог отзывов
m = df["reviews"].quantile(0.75)

# Формула Байеса
df["bayes_rating"] = ((df["reviews"] / (df["reviews"] + m)) * df["rating"] + (m / (df["reviews"] + m)) * C)

for i in ["bayes_rating", "rating"]:
    # Средний рейтинг по каждому уникальному ресторану
    rest_rating = df.groupby("name")[i].mean().reset_index()

    # TOP 10
    print("TOP 10")
    print(rest_rating.sort_values(i, ascending=False).head(10))

    # BOTTOM 10
    print("\nBOTTOM 10")
    print(rest_rating.sort_values(i, ascending=True).head(10))

df = df.drop(columns=["name", "reviews"])

corr_data = df[["avg_check", "hours", "WiFi(1/0)", "delivery(1/0)", "rating", "bayes_rating"]].corr().loc[["rating", "bayes_rating"],
              ["avg_check", "hours", "WiFi(1/0)", "delivery(1/0)"]]

plt.figure(figsize=(8, 3))

sns.heatmap(corr_data, annot=True, cmap="magma")

plt.ylabel("Рейтинги")
plt.xlabel("Признаки")

plt.show()

categ, num = ['cuisine','type','district','microdistict'], 0

while num < len(categ):
    com = 0

    for i in ["bayes_rating", "rating"]:
        groups = [group[i].values for _, group in df.groupby(categ[num])]
        f_stat, p_value = f_oneway(*groups)

        if p_value > 0.05:
            com += 1

        #df.groupby(categ[num])[i].mean().sort_values().plot(kind="barh")

        #plt.show()

    if com > 0:
        df = df.drop(columns=categ[num])
    num += 1

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()

    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        # Определение границ "усов"
        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR

        # Фильтрация данных
        initial_count = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

        print(f"Колонка '{col}': удалено {initial_count - len(df_clean)} строк(и).")

    return df_clean

# Список колонок с выбросами на ваших графиках
cols_to_fix = ['avg_check', 'rating', 'bayes_rating', 'hours']

# Применение функции (предположим, ваш DataFrame называется df)
df = remove_outliers_iqr(df, cols_to_fix)

df = pd.get_dummies(df, columns=['type', 'cuisine']).astype(int)

# Признаки
X = df.drop(["rating", "bayes_rating"], axis=1)

# Целевая переменная
y = df["rating"]

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=44
)

# Модели
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=44),
    "XGBoost": XGBRegressor(random_state=44),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=44)
}

results = []

# Обучение и оценка
for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "R2": r2
    })

# Таблица результатов
results_df = pd.DataFrame(results)

# Сортировка по R2
results_df = results_df.sort_values("R2", ascending=False)

print(results_df)