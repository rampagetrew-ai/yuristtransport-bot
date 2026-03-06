"""
Telegram RAG-бот: транспортное законодательство РФ
Поиск: Yandex Embeddings API (семантический)
Ответы: YandexGPT Pro
"""

import os
import json
import logging
import pickle
import asyncio
import time
import math
import re
import aiohttp
import numpy as np
from collections import Counter
from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
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

# Ищем файлы в корне репо, потом в data/ — работает при любом расположении
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

TOP_K             = 6
MIN_SCORE         = 0.3
MAX_CONTEXT_CHARS = 6000

# Yandex endpoints
EMBED_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
GPT_URL   = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

EMBED_MODEL_DOC   = f"emb://{YANDEX_FOLDER_ID}/text-search-doc/latest"
EMBED_MODEL_QUERY = f"emb://{YANDEX_FOLDER_ID}/text-search-query/latest"
GPT_MODEL         = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-pro/rc"

SYSTEM_PROMPT = (
    "Ты — юридический помощник по российскому транспортному законодательству. "
    "Отвечай ТОЛЬКО на основе предоставленных фрагментов нормативных актов. "
    "Если ответ не содержится в контексте — честно скажи об этом. "
    "Всегда указывай источник: название документа и номер статьи или пункта. "
    "Приводи конкретные цифры: штрафы, сроки, размеры ответственности. "
    "Отвечай на русском языке, структурированно и по существу."
)


# ── BM25 (fallback если нет индекса) ─────────────────────────────────────────
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


# ── Yandex API ────────────────────────────────────────────────────────────────
def yandex_headers() -> dict:
    return {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json",
    }


async def embed_one(session: aiohttp.ClientSession, text: str, doc_mode=True) -> list[float]:
    model = EMBED_MODEL_DOC if doc_mode else EMBED_MODEL_QUERY
    payload = {"modelUri": model, "text": text[:2000]}  # лимит Yandex
    for attempt in range(3):
        try:
            async with session.post(EMBED_URL, json=payload, headers=yandex_headers()) as r:
                if r.status == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                data = await r.json()
                return data["embedding"]
        except Exception as e:
            if attempt == 2:
                raise
            await asyncio.sleep(1)


async def gpt_answer(session: aiohttp.ClientSession, system: str, user: str) -> str:
    payload = {
        "modelUri": GPT_MODEL,
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 1500,
        },
        "messages": [
            {"role": "system", "text": system},
            {"role": "user",   "text": user},
        ],
    }
    async with session.post(GPT_URL, json=payload, headers=yandex_headers()) as r:
        r.raise_for_status()
        data = await r.json()
        return data["result"]["alternatives"][0]["message"]["text"]


# ── Index build (запускается один раз) ───────────────────────────────────────
async def build_index():
    log.info(f"Загружаю базу знаний: {KB_FILE}")
    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)
    chunks = kb["chunks"]
    total  = len(chunks)
    log.info(f"Чанков: {total} — начинаю эмбеддинг (это займёт ~20-30 мин)...")

    embeddings = []
    batch_size = 5   # Yandex rate limit ~10 rps

    async with aiohttp.ClientSession() as session:
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            tasks = [embed_one(session, c["text"], doc_mode=True) for c in batch]
            results = await asyncio.gather(*tasks)
            embeddings.extend(results)
            done = min(i + batch_size, total)
            log.info(f"  {done}/{total} ({done*100//total}%)")
            await asyncio.sleep(0.6)  # пауза между батчами

    emb_array = np.array(embeddings, dtype="float32")
    # L2-нормализация для cosine similarity через dot product
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    emb_array /= np.maximum(norms, 1e-8)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"embeddings": emb_array, "chunks": chunks}, f)

    log.info(f"✅ Индекс сохранён: {INDEX_FILE}  ({total} векторов)")


def load_index() -> tuple[np.ndarray, list[dict]]:
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    log.info(f"Индекс загружен: {len(data['chunks'])} векторов")
    return data["embeddings"], data["chunks"]


# ── Retrieval ─────────────────────────────────────────────────────────────────
async def retrieve_semantic(
    query: str,
    embeddings: np.ndarray,
    chunks: list[dict],
) -> list[dict]:
    """Семантический поиск через Yandex Embeddings."""
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


def retrieve_bm25(query: str, chunks: list[dict], bm25: BM25) -> list[dict]:
    """BM25-поиск как fallback."""
    results = bm25.search(query, top_k=TOP_K)
    out = []
    for idx, score in results:
        c = dict(chunks[idx])
        c["score"] = score / 20  # нормализуем в [0, 1]
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


# ── Telegram handlers ─────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Привет!* Я бот по транспортному законодательству РФ.\n\n"
        "📚 *База знаний (20 документов):*\n"
        "• КоАП — нарушения ПДД, транспорт, таможня\n"
        "• Штрафы ПДД (полная таблица)\n"
        "• УАТ — Устав автомобильного транспорта\n"
        "• Правила перевозки грузов и тяжеловесов\n"
        "• Правила ЭТРН и ЭДО\n"
        "• ФЗ-87, ПП №554 — транспортная экспедиция\n"
        "• ГК РФ — экспедиция и расчёты\n"
        "• УК РФ — транспортные преступления\n"
        "• НК РФ — НДС, акцизы, транспортный налог\n"
        "• Нормы рабочего времени водителей\n\n"
        "💬 *Просто задайте вопрос:*\n"
        "_— Штраф за превышение на 40 км/ч?_\n"
        "_— Ответственность экспедитора за утрату груза?_\n"
        "_— Нормы времени погрузки для 5-тонника?_\n\n"
        "/help — подробная справка",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ *Как пользоваться*\n\n"
        "Задайте вопрос на русском — бот найдёт ответ в нормативных актах "
        "и укажет источник (документ + статью).\n\n"
        "*Примеры:*\n"
        "• Штраф за езду без ОСАГО?\n"
        "• Нормы суточного пробега?\n"
        "• Когда нужна ТТН, а когда ЭТРН?\n"
        "• Порядок получения разрешения на тяжеловес?\n"
        "• Срок давности по договору экспедиции?\n"
        "• НДС 0% при международных перевозках — условия?\n"
        "• Ответственность за нарушение режима труда водителя?\n\n"
        "⚠️ Бот даёт справочную информацию. "
        "По конкретным ситуациям рекомендуем консультацию юриста.",
        parse_mode="Markdown",
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    if not query:
        return

    await update.message.chat.send_action("typing")

    use_semantic: bool      = context.bot_data.get("use_semantic", False)
    embeddings: np.ndarray  = context.bot_data.get("embeddings")
    chunks: list[dict]      = context.bot_data["chunks"]
    bm25: BM25              = context.bot_data["bm25"]

    try:
        # Поиск: семантический или BM25
        if use_semantic and embeddings is not None:
            relevant = await retrieve_semantic(query, embeddings, chunks)
        else:
            relevant = retrieve_bm25(query, chunks, bm25)

        if not relevant or relevant[0]["score"] < MIN_SCORE:
            await update.message.reply_text(
                "🤔 Не нашёл достаточно релевантной информации.\n"
                "Попробуйте переформулировать вопрос или задайте его конкретнее."
            )
            return

        ctx = build_context(relevant)
        user_msg = (
            f"Вопрос: {query}\n\n"
            f"Фрагменты нормативных актов:\n{ctx}\n\n"
            "Дай точный структурированный ответ. Укажи источник (документ и статью)."
        )

        async with aiohttp.ClientSession() as session:
            answer = await gpt_answer(session, SYSTEM_PROMPT, user_msg)

        # Источники
        seen, sources = set(), []
        for c in relevant[:4]:
            if c["title"] not in seen and c["score"] >= MIN_SCORE:
                seen.add(c["title"])
                sources.append(c["title"])

        search_tag = "🔍 семантический" if use_semantic else "🔍 BM25"
        footer = f"\n\n📌 *Источники ({search_tag}):*\n" + "\n".join(f"• {s}" for s in sources)
        full = answer + footer

        if len(full) > 4096:
            full = full[:4090] + "…"

        await update.message.reply_text(full, parse_mode="Markdown")

    except aiohttp.ClientResponseError as e:
        log.error(f"Yandex API error: {e.status} {e.message}")
        await update.message.reply_text("⚠️ Ошибка Yandex API. Попробуйте через минуту.")
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        await update.message.reply_text("⚠️ Внутренняя ошибка. Попробуйте ещё раз.")


# ── Startup ───────────────────────────────────────────────────────────────────
async def post_init(application):
    log.info("Загружаю базу знаний...")
    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)
    chunks = kb["chunks"]

    # BM25 — всегда строим (быстро, fallback)
    log.info(f"Строю BM25 индекс ({len(chunks)} чанков)...")
    bm25 = BM25([c["text"] for c in chunks])
    application.bot_data["chunks"] = chunks
    application.bot_data["bm25"]   = bm25

    # Semantic index — если уже создан
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
        log.info("ℹ️ embeddings_index.pkl не найден — используется BM25")
        log.info("   Запустите: python bot.py --build-index")
        application.bot_data["use_semantic"] = False

    await application.bot.set_my_commands([
        BotCommand("start", "О боте и примеры вопросов"),
        BotCommand("help",  "Как пользоваться"),
    ])
    log.info("✅ Бот готов к работе")


def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("🚀 Запуск polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true",
                        help="Создать embeddings_index.pkl через Yandex API и выйти")
    args = parser.parse_args()

    if args.build_index:
        asyncio.run(build_index())
    else:
        if not os.path.exists(INDEX_FILE):
            log.warning("embeddings_index.pkl не найден — бот стартует с BM25-поиском")
            log.warning("Для семантического поиска запустите: python bot.py --build-index")
        main()
