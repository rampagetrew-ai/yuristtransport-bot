#!/bin/bash
# deploy.sh — автодеплой Transport Law Bot на Timeweb Ubuntu 22.04
# Запуск: bash deploy.sh
set -e

REPO_URL="https://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@github.com/ВАШ_АККАУНТ/ВАШ_РЕПО.git"
APP_DIR="/opt/transport-bot"
SERVICE="transport-bot"

echo "========================================"
echo " Transport Law Bot — деплой на Timeweb"
echo "========================================"

echo "[1/6] Системные пакеты..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git

echo "[2/6] Клонирую / обновляю репозиторий..."
if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR" && git pull
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

echo "[3/6] Виртуальное окружение..."
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip -q
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

echo "[4/6] Создаю .env..."
cat > "$APP_DIR/.env" << 'ENVEOF'
TELEGRAM_TOKEN=8740777660:AAH6iGflf2vAsA2-RUvd5F0dDSZMM9prUU8
YANDEX_API_KEY=AQVN1EKR121OHtYrfMaUo3SMR14wBqfzdZ2HrmI8
YANDEX_FOLDER_ID=b1g1trtcj3fikcu105ak
ENVEOF
echo "  .env создан."

echo "[5/6] Systemd сервис..."
cp "$APP_DIR/transport-bot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable "$SERVICE"

echo "[6/6] Запускаю бота..."
systemctl restart "$SERVICE"
sleep 3
systemctl status "$SERVICE" --no-pager

echo ""
echo "✅ Готово! Бот запущен с BM25-поиском."
echo ""
echo "Для включения семантического поиска (Yandex Embeddings):"
echo "  cd $APP_DIR && source venv/bin/activate"
echo "  python bot.py --build-index   # ~20-30 минут"
echo "  systemctl restart $SERVICE"
echo ""
echo "Логи в реальном времени:"
echo "  journalctl -u $SERVICE -f"
