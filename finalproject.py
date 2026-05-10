import telebot
import pandas as pd

TOKEN = "8188334787:AAFwZq-EyaZA2CAh094xPh1MfD4k6K2h-7o"

bot = telebot.TeleBot(TOKEN)

# 📊 загружаем твои данные (из парсера)
df = pd.read_csv("bishkek_final.csv")

# -------------------------
# SMART FILTER
# -------------------------
def get_top(n=10):
    return df.sort_values(by="smart_score", ascending=False).head(n)


def search_cuisine(query):
    return df[df["cuisine"].fillna("").str.contains(query.lower())] \
        .sort_values(by="smart_score", ascending=False).head(10)


# -------------------------
# START
# -------------------------
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(
        message.chat.id,
        "🍽 Привет! Я бот по лучшим заведениям Бишкека.\n\n"
        "Команды:\n"
        "/top - лучшие места\n"
        "/cafe - кофейни\n"
        "/rest - рестораны\n"
        "/fastfood - фастфуд\n"
        "или просто напиши кухню (например: итальянская)"
    )


# -------------------------
# TOP
# -------------------------
@bot.message_handler(commands=['top'])
def top(message):
    top_places = get_top(10)

    text = "🏆 ТОП-10 заведений:\n\n"

    for _, row in top_places.iterrows():
        text += f"⭐ {row['name']}\n"
        text += f"Rating: {row['rating']} | Reviews: {row['reviews']}\n"
        text += f"Cuisine: {row.get('cuisine')}\n\n"

    bot.send_message(message.chat.id, text)


# -------------------------
# CAFÉ
# -------------------------
@bot.message_handler(commands=['cafe'])
def cafe(message):
    data = search_cuisine("коф")

    text = "☕ Лучшие кофейни:\n\n"

    for _, row in data.iterrows():
        text += f"⭐ {row['name']} ({row['rating']})\n"

    bot.send_message(message.chat.id, text)


# -------------------------
# RESTAURANTS
# -------------------------
@bot.message_handler(commands=['rest'])
def rest(message):
    data = search_cuisine("ресторан")

    text = "🍽 Лучшие рестораны:\n\n"

    for _, row in data.iterrows():
        text += f"⭐ {row['name']} ({row['rating']})\n"

    bot.send_message(message.chat.id, text)


# -------------------------
# FAST FOOD
# -------------------------
@bot.message_handler(commands=['fastfood'])
def fastfood(message):
    data = search_cuisine("фаст")

    text = "🍔 Фастфуд:\n\n"

    for _, row in data.iterrows():
        text += f"⭐ {row['name']} ({row['rating']})\n"

    bot.send_message(message.chat.id, text)


# -------------------------
# TEXT SEARCH
# -------------------------
@bot.message_handler(func=lambda message: True)
def text_search(message):
    query = message.text.lower()

    result = df[df["cuisine"].fillna("").str.lower().str.contains(query)]

    if len(result) == 0:
        bot.send_message(message.chat.id, "❌ Ничего не найдено")
        return

    result = result.sort_values(by="smart_score", ascending=False).head(10)

    text = f"🔍 Результаты по '{query}':\n\n"

    for _, row in result.iterrows():
        text += f"⭐ {row['name']} ({row['rating']})\n"

    bot.send_message(message.chat.id, text)


# -------------------------
# RUN
# -------------------------
print("🤖 Bot started...")
bot.polling()