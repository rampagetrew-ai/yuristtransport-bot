"""
Telegram RAG-бот: транспортное законодательство РФ
Поиск: Yandex Embeddings (семантический) / BM25 (fallback)
Ответы: DeepSeek API
Админ: @ALTLPU — статистика, блокировка пользователей
"""

import os
import json
import logging
import pickle
import asyncio
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime

import aiohttp
import numpy as np
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("transport_bot")

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
YANDEX_API_KEY   = os.environ["YANDEX_API_KEY"]
YANDEX_FOLDER_ID = os.environ["YANDEX_FOLDER_ID"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

ADMIN_USERNAME   = "ALTLPU"  # без @

DEEPSEEK_URL     = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL   = "deepseek-chat"

_BASE = os.path.dirname(os.path.abspath(__file__))
KB_FILE = (
    os.path.join(_BASE, "knowledge_base_transport.json")
    if os.path.exists(os.path.join(_BASE, "knowledge_base_transport.json"))
    else os.path.join(_BASE, "data", "knowledge_base_transport.json")
)
INDEX_FILE = (
    os.path.join(_BASE, "embeddings_index.pkl")
    if os.path.exists(os.path.join(_BASE, "embeddings_index.pkl"))
    else os.path.join(_BASE, "data", "embeddings_index.pkl")
)
STATS_FILE = os.path.join(_BASE, "data", "stats.json")
BLOCKED_FILE = os.path.join(_BASE, "data", "blocked.json")

TOP_K             = 6
MIN_SCORE         = 0.3
MAX_CONTEXT_CHARS = 5000

EMBED_URL         = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
EMBED_MODEL_DOC   = f"emb://{YANDEX_FOLDER_ID}/text-search-doc/latest"
EMBED_MODEL_QUERY = f"emb://{YANDEX_FOLDER_ID}/text-search-query/latest"

SYSTEM_PROMPT = """Ты — юридический помощник по российскому транспортному законодательству.

ПРАВИЛА ОТВЕТА:
1. Отвечай КОРОТКО и ЛАКОНИЧНО — максимум 3-5 предложений
2. ВСЕГДА указывай конкретную статью или пункт документа в формате: (ст.12.9 КоАП, п.26.1 ПДД, ст.796 ГК РФ и т.п.)
3. Приводи конкретные цифры: размеры штрафов, сроки, нормы
4. Если вопрос требует развёрнутого ответа — дай краткую выжимку главного
5. Отвечай ТОЛЬКО на основе предоставленных фрагментов
6. Если информации нет — честно скажи об этом одним предложением
7. Не используй вводные фразы типа "Согласно предоставленным данным..."
8. Отвечай на русском языке"""


# ── Хранилище статистики и блокировок ────────────────────────────────────────
def _ensure_data_dir():
    os.makedirs(os.path.join(_BASE, "data"), exist_ok=True)


def load_stats() -> dict:
    _ensure_data_dir()
    try:
        with open(STATS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"total_questions": 0, "users": {}}


def save_stats(stats: dict):
    _ensure_data_dir()
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def load_blocked() -> list:
    _ensure_data_dir()
    try:
        with open(BLOCKED_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_blocked(blocked: list):
    _ensure_data_dir()
    with open(BLOCKED_FILE, "w", encoding="utf-8") as f:
        json.dump(blocked, f, ensure_ascii=False, indent=2)


def record_question(user_id: int, username: str, question: str):
    stats = load_stats()
    stats["total_questions"] = stats.get("total_questions", 0) + 1
    uid = str(user_id)
    if uid not in stats["users"]:
        stats["users"][uid] = {
            "username": username,
            "questions": 0,
            "last_seen": "",
            "history": []
        }
    stats["users"][uid]["questions"] += 1
    stats["users"][uid]["last_seen"] = datetime.now().strftime("%d.%m.%Y %H:%M")
    stats["users"][uid]["username"] = username
    # Хранить последние 20 вопросов
    history = stats["users"][uid].get("history", [])
    history.append({"q": question[:100], "t": datetime.now().strftime("%d.%m %H:%M")})
    stats["users"][uid]["history"] = history[-20:]
    save_stats(stats)


# ── BM25 ──────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z]{2,}", text.lower())


class BM25:
    def __init__(self, corpus: list[str], k1=1.5, b=0.75):
        self.k1 = k1
        self.b  = b
        self.corpus_size = len(corpus)
        self.tokenized   = [tokenize(doc) for doc in corpus]
        self.avgdl       = sum(len(d) for d in self.tokenized) / max(self.corpus_size, 1)
        df: Counter = Counter()
        for doc in self.tokenized:
            for term in set(doc):
                df[term] += 1
        self.idf = {
            t: math.log((self.corpus_size - f + 0.5) / (f + 0.5) + 1)
            for t, f in df.items()
        }

    def search(self, query: str, top_k=10) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        scores = []
        for i, doc in enumerate(self.tokenized):
            tf_map = Counter(doc)
            dl = len(doc)
            s = 0.0
            for term in q_tokens:
                if term not in self.idf:
                    continue
                tf = tf_map[term]
                s += self.idf[term] * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scores[:top_k] if s > 0]


# ── Yandex Embeddings ─────────────────────────────────────────────────────────
def yandex_headers() -> dict:
    return {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json",
    }


async def embed_one(session: aiohttp.ClientSession, text: str, doc_mode=True) -> list[float]:
    model = EMBED_MODEL_DOC if doc_mode else EMBED_MODEL_QUERY
    payload = {"modelUri": model, "text": text[:2000]}
    for attempt in range(3):
        try:
            async with session.post(EMBED_URL, json=payload, headers=yandex_headers()) as r:
                if r.status == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                data = await r.json()
                return data["embedding"]
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(1)


async def build_index():
    log.info(f"Загружаю базу знаний: {KB_FILE}")
    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)
    chunks = kb["chunks"]
    total  = len(chunks)
    log.info(f"Чанков: {total} — начинаю эмбеддинг (~20-30 мин)...")

    embeddings = []
    batch_size = 5
    async with aiohttp.ClientSession() as session:
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            tasks = [embed_one(session, c["text"], doc_mode=True) for c in batch]
            results = await asyncio.gather(*tasks)
            embeddings.extend(results)
            done = min(i + batch_size, total)
            log.info(f"  {done}/{total} ({done*100//total}%)")
            await asyncio.sleep(0.6)

    emb_array = np.array(embeddings, dtype="float32")
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    emb_array /= np.maximum(norms, 1e-8)

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"embeddings": emb_array, "chunks": chunks}, f)
    log.info(f"✅ Индекс сохранён: {INDEX_FILE} ({total} векторов)")


def load_index() -> tuple[np.ndarray, list[dict]]:
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    log.info(f"Индекс загружен: {len(data['chunks'])} векторов")
    return data["embeddings"], data["chunks"]


# ── Retrieval ─────────────────────────────────────────────────────────────────
async def retrieve_semantic(query, embeddings, chunks) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        q_emb = await embed_one(session, query, doc_mode=False)
    q_vec = np.array(q_emb, dtype="float32")
    q_vec /= max(np.linalg.norm(q_vec), 1e-8)
    scores  = embeddings @ q_vec
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    results = []
    for idx in top_idx:
        c = dict(chunks[idx])
        c["score"] = float(scores[idx])
        results.append(c)
    return results


def retrieve_bm25(query, chunks, bm25) -> list[dict]:
    results = bm25.search(query, top_k=TOP_K)
    out = []
    for idx, score in results:
        c = dict(chunks[idx])
        c["score"] = score / 20
        out.append(c)
    return out


def build_context(relevant: list[dict]) -> str:
    parts, total = [], 0
    for c in relevant:
        block = f"[{c['title']}]\n{c['text']}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


# ── DeepSeek ──────────────────────────────────────────────────────────────────
async def deepseek_answer(session: aiohttp.ClientSession, system: str, user: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": 600,
    }
    async with session.post(DEEPSEEK_URL, json=payload, headers=headers) as r:
        r.raise_for_status()
        data = await r.json()
        return data["choices"][0]["message"]["content"]


# ── Анимация "ответ формируется" ──────────────────────────────────────────────
async def animate_thinking(message, stop_event: asyncio.Event):
    frames = [
        "⏳ Ищу в базе законов...",
        "📖 Анализирую нормативные акты...",
        "⚖️ Формирую ответ...",
        "📝 Готовлю ссылки на статьи...",
    ]
    i = 0
    while not stop_event.is_set():
        try:
            await message.edit_text(frames[i % len(frames)])
        except Exception:
            pass
        i += 1
        await asyncio.sleep(2)


# ── Проверки ──────────────────────────────────────────────────────────────────
def is_admin(update: Update) -> bool:
    username = update.effective_user.username or ""
    return username.lower() == ADMIN_USERNAME.lower()


def is_blocked(user_id: int) -> bool:
    blocked = load_blocked()
    return user_id in blocked


# ── Команды пользователя ──────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Привет!* Я бот по транспортному законодательству РФ.\n\n"
        "📚 *База знаний (21 документ):*\n"
        "• ПДД РФ 2026 (полный текст)\n"
        "• КоАП — нарушения ПДД, транспорт, таможня\n"
        "• Штрафы ПДД (полная таблица)\n"
        "• УАТ — Устав автомобильного транспорта\n"
        "• Правила перевозки грузов и тяжеловесов\n"
        "• ЭТРН, ФЗ-87, ГК, УК, НК — транспортные разделы\n"
        "• Нормы рабочего времени водителей\n\n"
        "💬 *Задайте вопрос:*\n"
        "_— Штраф за превышение на 40 км/ч?_\n"
        "_— Сколько часов водитель может ехать без отдыха?_\n"
        "_— Ответственность экспедитора за утрату груза?_",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ *Как пользоваться*\n\n"
        "Задайте вопрос — получите краткий ответ со ссылкой на статью.\n\n"
        "*Примеры:*\n"
        "• Штраф за езду без ОСАГО?\n"
        "• Норма непрерывного вождения?\n"
        "• Когда нужна ТТН, а когда ЭТРН?\n"
        "• НДС 0% при международных перевозках?\n"
        "• Срок давности по договору экспедиции?\n\n"
        "⚠️ Бот даёт справочную информацию. По конкретным ситуациям рекомендуем консультацию юриста.",
        parse_mode="Markdown",
    )


# ── Основной обработчик сообщений ─────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user    = update.effective_user
    query   = update.message.text.strip()
    if not query:
        return

    # Проверка блокировки
    if is_blocked(user.id):
        await update.message.reply_text("🚫 Вы заблокированы.")
        return

    # Записать статистику
    record_question(user.id, user.username or str(user.id), query)

    # Отправить анимацию
    thinking_msg = await update.message.reply_text("⏳ Ищу в базе законов...")
    stop_event   = asyncio.Event()
    anim_task    = asyncio.create_task(animate_thinking(thinking_msg, stop_event))

    chunks         = context.bot_data["chunks"]
    bm25           = context.bot_data["bm25"]
    use_semantic   = context.bot_data.get("use_semantic", False)
    embeddings     = context.bot_data.get("embeddings")

    try:
        # Поиск
        if use_semantic and embeddings is not None:
            relevant = await retrieve_semantic(query, embeddings, chunks)
        else:
            relevant = retrieve_bm25(query, chunks, bm25)

        if not relevant or relevant[0]["score"] < MIN_SCORE:
            stop_event.set()
            await anim_task
            await thinking_msg.edit_text(
                "🤔 Не нашёл информации по этому вопросу в базе законов.\n"
                "Попробуйте переформулировать."
            )
            return

        ctx = build_context(relevant)
        user_msg = (
            f"Вопрос: {query}\n\n"
            f"Фрагменты нормативных актов:\n{ctx}"
        )

        async with aiohttp.ClientSession() as session:
            answer = await deepseek_answer(session, SYSTEM_PROMPT, user_msg)

        # Источники (уникальные, с оценкой >= MIN_SCORE)
        seen, sources = set(), []
        for c in relevant[:4]:
            if c["title"] not in seen and c["score"] >= MIN_SCORE:
                seen.add(c["title"])
                sources.append(c["title"])

        footer = "\n\n📌 *Источники:*\n" + "\n".join(f"• {s}" for s in sources)
        full   = answer + footer

        if len(full) > 4096:
            full = full[:4090] + "…"

        stop_event.set()
        await anim_task
        await thinking_msg.edit_text(full, parse_mode="Markdown")

    except aiohttp.ClientResponseError as e:
        stop_event.set()
        await anim_task
        log.error(f"DeepSeek API error: {e.status}")
        await thinking_msg.edit_text("⚠️ Ошибка сервиса. Попробуйте через минуту.")
    except Exception as e:
        stop_event.set()
        await anim_task
        log.exception(f"Unexpected error: {e}")
        await thinking_msg.edit_text("⚠️ Внутренняя ошибка. Попробуйте ещё раз.")


# ── Админ-команды ─────────────────────────────────────────────────────────────
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return

    stats   = load_stats()
    blocked = load_blocked()

    text = (
        f"🔧 *Панель администратора*\n\n"
        f"👥 Всего пользователей: {len(stats['users'])}\n"
        f"❓ Всего вопросов: {stats.get('total_questions', 0)}\n"
        f"🚫 Заблокировано: {len(blocked)}\n\n"
        f"*Команды:*\n"
        f"/stats — подробная статистика\n"
        f"/user @username — статистика по пользователю\n"
        f"/block @username — заблокировать\n"
        f"/unblock @username — разблокировать\n"
        f"/blocked — список заблокированных"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return

    stats = load_stats()
    users = stats.get("users", {})

    # Топ-10 по вопросам
    top = sorted(users.items(), key=lambda x: x[1].get("questions", 0), reverse=True)[:10]

    lines = [f"📊 *Статистика бота*\n", f"Всего вопросов: {stats.get('total_questions', 0)}", f"Всего пользователей: {len(users)}\n", "*Топ-10 активных:*"]
    for uid, u in top:
        uname = u.get("username", uid)
        q     = u.get("questions", 0)
        last  = u.get("last_seen", "—")
        lines.append(f"• @{uname}: {q} вопр. (посл: {last})")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return

    if not context.args:
        await update.message.reply_text("Использование: /user @username")
        return

    target = context.args[0].lstrip("@").lower()
    stats  = load_stats()

    found = None
    for uid, u in stats.get("users", {}).items():
        if (u.get("username") or "").lower() == target:
            found = (uid, u)
            break

    if not found:
        await update.message.reply_text(f"Пользователь @{target} не найден.")
        return

    uid, u = found
    history = u.get("history", [])
    history_text = "\n".join(f"  {h['t']}: {h['q']}" for h in history[-10:]) or "  нет"

    text = (
        f"👤 *@{u.get('username', uid)}*\n"
        f"ID: `{uid}`\n"
        f"Вопросов: {u.get('questions', 0)}\n"
        f"Последний визит: {u.get('last_seen', '—')}\n\n"
        f"*Последние вопросы:*\n{history_text}"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_block(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return

    if not context.args:
        await update.message.reply_text("Использование: /block @username или /block user_id")
        return

    target  = context.args[0].lstrip("@").lower()
    stats   = load_stats()
    blocked = load_blocked()

    # Найти user_id по username
    found_id = None
    for uid, u in stats.get("users", {}).items():
        if (u.get("username") or "").lower() == target:
            found_id = int(uid)
            break

    # Если не нашли по username — попробовать как числовой ID
    if not found_id:
        try:
            found_id = int(target)
        except ValueError:
            await update.message.reply_text(f"Пользователь @{target} не найден в базе.")
            return

    if found_id in blocked:
        await update.message.reply_text(f"Пользователь уже заблокирован.")
        return

    blocked.append(found_id)
    save_blocked(blocked)
    await update.message.reply_text(f"🚫 Пользователь {target} заблокирован (ID: {found_id}).")


async def cmd_unblock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return

    if not context.args:
        await update.message.reply_text("Использование: /unblock @username или /unblock user_id")
        return

    target  = context.args[0].lstrip("@").lower()
    stats   = load_stats()
    blocked = load_blocked()

    found_id = None
    for uid, u in stats.get("users", {}).items():
        if (u.get("username") or "").lower() == target:
            found_id = int(uid)
            break
    if not found_id:
        try:
            found_id = int(target)
        except ValueError:
            await update.message.reply_text(f"Пользователь @{target} не найден.")
            return

    if found_id not in blocked:
        await update.message.reply_text("Пользователь не заблокирован.")
        return

    blocked.remove(found_id)
    save_blocked(blocked)
    await update.message.reply_text(f"✅ Пользователь {target} разблокирован.")


async def cmd_blocked(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return

    blocked = load_blocked()
    stats   = load_stats()

    if not blocked:
        await update.message.reply_text("Список заблокированных пуст.")
        return

    lines = ["🚫 *Заблокированные пользователи:*\n"]
    for uid in blocked:
        u = stats.get("users", {}).get(str(uid), {})
        uname = u.get("username", str(uid))
        lines.append(f"• @{uname} (ID: {uid})")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ── Startup ───────────────────────────────────────────────────────────────────
async def post_init(application):
    log.info("Загружаю базу знаний...")
    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)
    chunks = kb["chunks"]

    log.info(f"Строю BM25 индекс ({len(chunks)} чанков)...")
    bm25 = BM25([c["text"] for c in chunks])
    application.bot_data["chunks"] = chunks
    application.bot_data["bm25"]   = bm25

    if os.path.exists(INDEX_FILE):
        try:
            embeddings, _ = load_index()
            application.bot_data["embeddings"]   = embeddings
            application.bot_data["use_semantic"] = True
            log.info("✅ Семантический поиск активен (Yandex Embeddings)")
        except Exception as e:
            log.warning(f"Не удалось загрузить индекс: {e} — используется BM25")
            application.bot_data["use_semantic"] = False
    else:
        log.info("ℹ️ embeddings_index.pkl не найден — BM25")
        application.bot_data["use_semantic"] = False

    await application.bot.set_my_commands([
        BotCommand("start",   "О боте и примеры вопросов"),
        BotCommand("help",    "Как пользоваться"),
        BotCommand("admin",   "Панель администратора"),
        BotCommand("stats",   "Статистика (только админ)"),
        BotCommand("user",    "Инфо о пользователе (только админ)"),
        BotCommand("block",   "Заблокировать пользователя (только админ)"),
        BotCommand("unblock", "Разблокировать (только админ)"),
        BotCommand("blocked", "Список заблокированных (только админ)"),
    ])
    log.info("✅ Бот готов к работе")


def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("admin",    cmd_admin))
    app.add_handler(CommandHandler("stats",    cmd_stats))
    app.add_handler(CommandHandler("user",     cmd_user))
    app.add_handler(CommandHandler("block",    cmd_block))
    app.add_handler(CommandHandler("unblock",  cmd_unblock))
    app.add_handler(CommandHandler("blocked",  cmd_blocked))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("🚀 Запуск polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true")
    args = parser.parse_args()

    if args.build_index:
        asyncio.run(build_index())
    else:
        main()
