"""Простой тестовый Telegram-бот на чистом Python (без зависимостей).

Запуск:  BOT_TOKEN=<токен> python3 bot.py
"""
import json
import os
import random
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

TOKEN = os.environ.get("BOT_TOKEN")
if not TOKEN:
    raise SystemExit("Укажите токен в переменной окружения BOT_TOKEN")
API = f"https://api.telegram.org/bot{TOKEN}/"

HELP = (
    "Привет! Я тестовый бот 🤖\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/help — список команд\n"
    "/time — текущее время (UTC)\n"
    "/dice — бросить кубик\n"
    "/coin — подбросить монетку\n"
    "Любой другой текст я просто повторю."
)


def call(method, **params):
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(API + method, data=data, timeout=60) as resp:
        return json.load(resp)


def reply(chat_id, text):
    call("sendMessage", chat_id=chat_id, text=text)


def handle(message):
    chat_id = message["chat"]["id"]
    text = message.get("text", "")
    cmd = text.split()[0].split("@")[0].lower() if text else ""

    if cmd in ("/start", "/help"):
        reply(chat_id, HELP)
    elif cmd == "/time":
        reply(chat_id, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    elif cmd == "/dice":
        call("sendDice", chat_id=chat_id)
    elif cmd == "/coin":
        reply(chat_id, random.choice(["Орёл 🦅", "Решка 🪙"]))
    elif text:
        reply(chat_id, f"Ты написал: {text}")
    else:
        reply(chat_id, "Я понимаю только текст 🙂")


def main():
    me = call("getMe")["result"]
    print(f"Бот @{me['username']} запущен", flush=True)
    offset = 0
    while True:
        try:
            updates = call("getUpdates", offset=offset, timeout=50)["result"]
            for upd in updates:
                offset = upd["update_id"] + 1
                if "message" in upd:
                    handle(upd["message"])
        except Exception as e:  # сеть/таймаут — подождать и продолжить
            print("Ошибка:", e, flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
