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
import numpy as np

df = pd.read_csv("restaurants.csv")

plt.figure(figsize=(12, 6))
order = df.groupby('cuisine')['rating'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='cuisine', y='rating', order=order, palette='viridis')
plt.xticks(rotation=45)
plt.title('Распределение рейтинга по типам кухни (от высшего к низшему)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='reviews', y='rating', hue='avg_check', palette='coolwarm', alpha=0.7)
plt.title('Взаимосвязь количества отзывов и рейтинга')
plt.xlabel('Количество отзывов')
plt.ylabel('Рейтинг')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='cuisine', order=df['cuisine'].value_counts().index, palette='magma')
plt.xticks(rotation=45)
plt.title('Количество заведений по типам кухни в датасете')
plt.show()

plt.figure(figsize=(8, 10))
district_pivot = df.groupby('district')[['rating', 'avg_check', 'hours']].mean()
sns.heatmap(district_pivot, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=.5)
plt.title('Тепловая карта: Средние показатели по районам')
plt.show()

C, m = df["rating"].mean(), df["reviews"].quantile(0.75)
df["bayes_rating"] = ((df["reviews"] / (df["reviews"] + m)) * df["rating"] + (m / (df["reviews"] + m)) * C)

for i in ["bayes_rating", "rating"]:
    rest_rating = df.groupby("name")[i].mean().reset_index()
    print("TOP 10")
    print(rest_rating.sort_values(i, ascending=False).head(10))
    print("\nBOTTOM 10")
    print(rest_rating.sort_values(i, ascending=True).head(10))

df = df.drop(columns=["name"])
corr_data = df[["avg_check", "hours", "WiFi(1/0)", "delivery(1/0)", "rating", "bayes_rating"]].corr().loc[["rating", "bayes_rating"],
              ["avg_check", "hours", "WiFi(1/0)", "delivery(1/0)"]]
categ, num = ['cuisine','type','district','microdistict'], 0
while num < len(categ):
    com = 0
    for i in ["bayes_rating", "rating"]:
        groups = [group[i].values for _, group in df.groupby(categ[num])]
        f_stat, p_value = f_oneway(*groups)
        if p_value > 0.05:
            com += 1
    if com > 0:
        df = df.drop(columns=categ[num])
    num += 1

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

cols_to_fix = ['avg_check', 'rating', 'bayes_rating', 'hours']
df = remove_outliers_iqr(df, cols_to_fix)
df = pd.get_dummies(df, columns=['type', 'cuisine']).astype(int)
df = df.drop(columns="reviews")

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["rating", "bayes_rating"], axis=1),
    df["rating"],
    test_size=0.2,
    random_state=42)
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results.append({"Model": name,
                "MAE": mae,"R2": r2})

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("R2", ascending=False)
print(results_df)
