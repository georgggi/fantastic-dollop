import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

df = pd.read_csv("restaurant1.csv")

# Средний рейтинг по всему датасету
C = df["rating"].mean()

# Минимальный порог отзывов
m = df["reviews"].quantile(0.75)

# Формула Байеса
df["bayes_rating"] = (
    (df["reviews"] / (df["reviews"] + m)) * df["rating"]
    +
    (m / (df["reviews"] + m)) * C
)

numeric_df = df.select_dtypes(include="number")

corr = numeric_df.corr()

print(corr)

sns.heatmap(df[["avg_check","hours","WiFi(1/0)","delivery(1/0)","bayes_rating"]].corr()[["bayes_rating"]].drop("bayes_rating").T, annot=True)

plt.show()
# Сортировка
df = df.sort_values("bayes_rating", ascending=False)

# Топ ресторанов
print(df[["name", "rating", "reviews", "bayes_rating"]].head(10))

plt.figure(figsize=(8,6))

sns.heatmap(corr, annot=True)

plt.show()

groups = [group["bayes_rating"].values for _, group in df.groupby("district")]

f_stat, p_value = f_oneway(*groups)

print(f_stat, p_value)

groups = [group["rating"].values for _, group in df.groupby("district")]

f_stat, p_value = f_oneway(*groups)

print(f_stat, p_value)

df.groupby("district")["bayes_rating"].mean().sort_values().plot(kind="barh")

plt.show()