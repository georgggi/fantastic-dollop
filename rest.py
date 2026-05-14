import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

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

corr_data = df[["avg_check", "hours", "WiFi(1/0)", "delivery(1/0)", "rating", "bayes_rating"]].corr().loc[["rating", "bayes_rating"],
              ["avg_check", "hours", "WiFi(1/0)", "delivery(1/0)"]]

plt.figure(figsize=(8, 3))

sns.heatmap(corr_data, annot=True, cmap="magma")

plt.ylabel("Рейтинги")
plt.xlabel("Признаки")

plt.show()

# Сортировка
df = df.sort_values("bayes_rating", ascending=False)

# Топ ресторанов
print(df[["name", "rating", "reviews", "bayes_rating"]].head(10))

print()

categ, num = ['name', 'cuisine','type','district','microdistict'], 0

while num < len(categ):

    print(categ[num].capitalize())

    for i in ["bayes_rating", "rating"]:

        print(i)

        groups = [group[i].values for _, group in df.groupby(categ[num])]

        f_stat, p_value = f_oneway(*groups)

        print(f'F: {f_stat}, P: {p_value}')

        df.groupby(categ[num])[i].mean().sort_values().plot(kind="barh")

        plt.show()

    num += 1
