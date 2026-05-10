import requests
import pandas as pd
import time

API_KEY = "b6572d78-e44e-41eb-b6a6-41598cbf7eef"

SEARCH_URL = "https://catalog.api.2gis.com/3.0/items"
DETAIL_URL = "https://catalog.api.2gis.com/3.0/items/byid"

queries = ["кафе", "ресторан", "кофейня", "фастфуд"]

seen_ids = set()
cache = {}
data = []


# -------------------------
# SEARCH
# -------------------------
def search(q, page):
    params = {
        "q": q,
        "location": "74.6122,42.8820",
        "radius": 5000,
        "page": page,
        "page_size": 20,
        "fields": "items.id,items.name,items.full_address_name",
        "key": API_KEY
    }

    r = requests.get(SEARCH_URL, params=params)
    return r.json()


# -------------------------
# DETAILS API
# -------------------------
def get_details(item_id):

    if item_id in cache:
        return cache[item_id]

    params = {
        "id": item_id,
        "fields": "rating,reviews,attributes,schedule,description",
        "key": API_KEY
    }

    r = requests.get(DETAIL_URL, params=params)
    result = r.json()

    cache[item_id] = result
    time.sleep(0.1)

    return result


# -------------------------
# SMART SCORE
# -------------------------
def smart_score(rating, reviews):
    if not rating:
        return 0

    reviews = reviews or 0
    weight = min(reviews / 100, 1)

    return rating * (0.5 + 0.5 * weight)


# -------------------------
# ATTRIBUTES PARSER
# -------------------------
def parse_attributes(item):
    cuisine = None
    wifi = False
    price = None

    for attr in item.get("attributes", []):
        text = (str(attr.get("name", "")) + " " + str(attr.get("value", ""))).lower()

        if "кухн" in text:
            cuisine = attr.get("value")

        if "wifi" in text or "wi-fi" in text:
            wifi = True

        if "чек" in text or "сом" in text:
            price = attr.get("value")

    desc = str(item.get("description", "")).lower()

    if not cuisine:
        if "итальян" in desc:
            cuisine = "итальянская"
        elif "азиат" in desc:
            cuisine = "азиатская"
        elif "восточ" in desc:
            cuisine = "восточная"

    if not wifi and "wifi" in desc:
        wifi = True

    return cuisine, wifi, price


# -------------------------
# MAIN LOOP
# -------------------------
for q in queries:
    print(f"\n🔍 {q}")

    for page in range(1, 6):

        result = search(q, page)
        items = result.get("result", {}).get("items", [])

        if not items:
            break

        for item in items:
            item_id = item.get("id")

            if item_id in seen_ids:
                continue

            seen_ids.add(item_id)

            # DETAILS
            details = get_details(item_id)
            full = details.get("result", {})

            rating_data = full.get("rating")
            reviews_data = full.get("reviews", {})

            # rating fix
            rating = None
            if isinstance(rating_data, dict):
                rating = rating_data.get("value")
            else:
                rating = rating_data

            reviews = reviews_data.get("count", 0)

            cuisine, wifi, price = parse_attributes(full)

            score = smart_score(rating, reviews)

            data.append({
                "name": item.get("name"),
                "address": item.get("full_address_name"),
                "rating": rating,
                "reviews": reviews,
                "smart_score": score,
                "cuisine": cuisine,
                "wifi": wifi,
                "price_range": price,
                "schedule": full.get("schedule", {}).get("text"),
            })

        print(f"  page {page}: +{len(items)}")


# -------------------------
# DATAFRAME
# -------------------------
df = pd.DataFrame(data)

print("\n✅ Всего заведений:", len(df))


# -------------------------
# CLEAN DATA
# -------------------------
df = df[
    (df["rating"].notna()) &
    (df["rating"] >= 3.5) &
    (df["reviews"] >= 3)
]
# -------------------------
# TOP BY SMART SCORE
# -------------------------
df_top = df.sort_values(by="smart_score", ascending=False)

print("\n🏆 ТОП-10 ЛУЧШИХ ЗАВЕДЕНИЙ:\n")

print(df_top[[
    "name",
    "rating",
    "reviews",
    "smart_score",
    "cuisine"
]].head(10))


# -------------------------
# TOP BY CUISINE
# -------------------------
print("\n🍽 ТОП ПО КУХНЕ:")

for cuisine in df["cuisine"].dropna().unique():

    print(f"\n--- {cuisine.upper()} ---")

    top = df[df["cuisine"] == cuisine].sort_values(
        by="smart_score",
        ascending=False
    ).head(5)

    print(top[[
        "name",
        "rating",
        "reviews",
        "smart_score"
    ]])


# -------------------------
# SAVE
# -------------------------
df.to_csv("bishkek_final.csv", index=False, encoding="utf-8-sig")
df.to_excel("bishkek_final.xlsx", index=False)

print("\n💾 Готово: CSV + Excel сохранены")