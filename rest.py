import pandas as pd

df = pd.read_csv('restaurants.csv')

df.loc[df['cuisine'] == 'Кыргызская', 'cuisine'] = 'Национальная'

df.to_csv('restaurant1.csv',
        index=False,
        encoding="utf-8-sig")
print(df)