# 🚛 Transport Law Bot

Telegram-бот по транспортному законодательству РФ.
Поиск: BM25 (сразу) → Yandex Embeddings (после build-index).
Ответы: YandexGPT Pro.

## Структура

```
├── bot.py
├── requirements.txt
├── deploy.sh                  ← запустить на сервере
├── transport-bot.service
├── .env.example
└── data/
    ├── knowledge_base_transport.json   (4678 чанков)
    └── chunks_for_embedding.json
```

## Деплой на Timeweb

```bash
ssh root@ВАШ_IP
curl -o deploy.sh https://raw.githubusercontent.com/ВАШ_АККАУНТ/ВАШ_РЕПО/main/deploy.sh
bash deploy.sh
```

## Команды на сервере

```bash
journalctl -u transport-bot -f        # логи
systemctl restart transport-bot        # перезапуск
systemctl stop transport-bot           # остановка

# Включить семантический поиск (один раз, ~20-30 мин):
cd /opt/transport-bot && source venv/bin/activate
python bot.py --build-index
systemctl restart transport-bot
```



- Telegram: @BotFather → /mybots → Revoke token
- Yandex: console.yandex.cloud → Сервисные аккаунты → Ключи → Удалить и создать новый
