"""
Telegram RAG-бот: транспортное законодательство РФ
Поиск: Yandex Embeddings (семантический) / BM25 (fallback + hybrid)
Ответы: DeepSeek API (fallback → YandexGPT)

Исправления:
  [4]  Админ по user_id вместо username
  [3]  Rate limiting — 1 запрос / 15 сек на пользователя
  [5]  Ограничение длины входящего сообщения
  [1]  asyncio.Lock + атомарная запись stats и blocked
  [18] Fallback на YandexGPT если DeepSeek недоступен
  [12] Hybrid search: embeddings + BM25 без hard-cut по порогу
  [13] Длинные ответы разбиваются на части
  [10] Контекст диалога — последние 3 обмена на chat_id
  [6]  numpy.save/load вместо pickle для эмбеддингов
  [A]  TTL-очистка _last_request — записи старше 1 часа удаляются автоматически
  [B]  Ограничение dialogs до 1000 чатов (LRU через OrderedDict)
  [C]  Flush незавершённых record_question тасков в post_stop
"""

import os
import json
import logging
import asyncio
import math
import re
import time
from collections import Counter, OrderedDict
from datetime import datetime
from logging.handlers import RotatingFileHandler

import aiohttp
import numpy as np
from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ── Пути ──────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_BASE, "data"), exist_ok=True)

# ── Logging с ротацией ────────────────────────────────────────────────────────
_fmt     = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh      = RotatingFileHandler(
    os.path.join(_BASE, "data", "bot.log"),
    maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_fh, _sh])
log = logging.getLogger("transport_bot")

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
YANDEX_API_KEY   = os.environ["YANDEX_API_KEY"]
YANDEX_FOLDER_ID = os.environ["YANDEX_FOLDER_ID"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

# [4] Идентификация по user_id — username можно сменить в любой момент
ADMIN_USER_ID = 7024585516

# [3] Rate limiting
RATE_LIMIT_SEC = 15  # секунд между запросами одного пользователя

# [5] Ограничение входящего текста
MAX_QUERY_LEN = 1000

DEEPSEEK_URL   = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# [18] YandexGPT как fallback
YANDEX_GPT_URL   = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEX_GPT_MODEL = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest"

KB_FILE = (
    os.path.join(_BASE, "knowledge_base_transport.json")
    if os.path.exists(os.path.join(_BASE, "knowledge_base_transport.json"))
    else os.path.join(_BASE, "data", "knowledge_base_transport.json")
)

# [6] numpy-формат вместо pickle
INDEX_NPY_FILE  = os.path.join(_BASE, "data", "embeddings.npy")
INDEX_META_FILE = os.path.join(_BASE, "data", "embeddings_meta.json")
# Обратная совместимость со старым pickle
INDEX_PKL_FILE = (
    os.path.join(_BASE, "embeddings_index.pkl")
    if os.path.exists(os.path.join(_BASE, "embeddings_index.pkl"))
    else os.path.join(_BASE, "data", "embeddings_index.pkl")
)

STATS_FILE   = os.path.join(_BASE, "data", "stats.json")
BLOCKED_FILE = os.path.join(_BASE, "data", "blocked.json")

TOP_K             = 8     # берём больше для hybrid-слияния
MIN_SCORE_SEM     = 0.25  # [12] снижен порог семантики
MAX_CONTEXT_CHARS = 5000

# [10] Контекст диалога
DIALOG_HISTORY_LEN = 3  # последних пар Q+A

EMBED_URL         = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
EMBED_MODEL_DOC   = f"emb://{YANDEX_FOLDER_ID}/text-search-doc/latest"
EMBED_MODEL_QUERY = f"emb://{YANDEX_FOLDER_ID}/text-search-query/latest"

SYSTEM_PROMPT = """Ты — юридический помощник по российскому транспортному законодательству.

ПРАВИЛА ОТВЕТА:
1. Отвечай КОРОТКО и ЛАКОНИЧНО — максимум 3-5 предложений
2. ВСЕГДА указывай конкретную статью или пункт: (ст.12.9 КоАП, п.26.1 ПДД, ст.796 ГК РФ)
3. Приводи конкретные цифры: штрафы, сроки, нормы
4. Отвечай ТОЛЬКО на основе предоставленных фрагментов нормативных актов
5. Если информации нет — честно скажи об этом одним предложением
6. Не используй вводные фразы типа "Согласно предоставленным данным..."
7. Отвечай на русском языке"""


# ── [1] Locks для безопасной записи файлов ────────────────────────────────────
_stats_lock   = asyncio.Lock()
_blocked_lock = asyncio.Lock()


# ── Статистика ────────────────────────────────────────────────────────────────
def load_stats() -> dict:
    try:
        with open(STATS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"total_questions": 0, "users": {}}


def _write_stats(stats: dict):
    """Атомарная запись: пишем во tmp, потом rename."""
    tmp = STATS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATS_FILE)


async def record_question(user_id: int, username: str, question: str):
    async with _stats_lock:
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
        stats["users"][uid]["last_seen"]  = datetime.now().strftime("%d.%m.%Y %H:%M")
        stats["users"][uid]["username"]   = username or str(user_id)
        history = stats["users"][uid].get("history", [])
        history.append({"q": question[:100], "t": datetime.now().strftime("%d.%m %H:%M")})
        stats["users"][uid]["history"] = history[-20:]
        await asyncio.get_event_loop().run_in_executor(None, _write_stats, stats)


# ── Блокировки ────────────────────────────────────────────────────────────────
def load_blocked() -> list:
    try:
        with open(BLOCKED_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _write_blocked(blocked: list):
    tmp = BLOCKED_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(blocked, f, ensure_ascii=False, indent=2)
    os.replace(tmp, BLOCKED_FILE)


async def save_blocked(blocked: list):
    async with _blocked_lock:
        await asyncio.get_event_loop().run_in_executor(None, _write_blocked, blocked)


def is_blocked(user_id: int) -> bool:
    return user_id in load_blocked()


# ── [3] Rate limiting ─────────────────────────────────────────────────────────
_last_request: dict[int, float] = {}
_RATE_TTL = 3600  # [A] записи старше 1 часа не нужны — удаляем


def check_rate_limit(user_id: int) -> float:
    """Возвращает 0 если можно отвечать, иначе — сколько секунд ждать.
    [A] Попутно удаляет записи пользователей, неактивных больше часа.
    """
    now  = time.monotonic()

    # [A] TTL-очистка: удаляем все записи старше _RATE_TTL за один проход
    stale = [uid for uid, ts in _last_request.items() if now - ts > _RATE_TTL]
    for uid in stale:
        del _last_request[uid]

    last = _last_request.get(user_id, 0)
    diff = now - last
    if diff < RATE_LIMIT_SEC:
        return RATE_LIMIT_SEC - diff
    _last_request[user_id] = now
    return 0


# ── BM25 ──────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z]{2,}", text.lower())


class BM25:
    def __init__(self, corpus: list[str], k1=1.5, b=0.75):
        self.k1 = k1
        self.b  = b
        self.corpus_size = len(corpus)
        self.tokenized   = [tokenize(doc) for doc in corpus]
        self.avgdl = sum(len(d) for d in self.tokenized) / max(self.corpus_size, 1)
        df: Counter = Counter()
        for doc in self.tokenized:
            for term in set(doc):
                df[term] += 1
        self.idf = {
            t: math.log((self.corpus_size - f + 0.5) / (f + 0.5) + 1)
            for t, f in df.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        scores = []
        for i, doc in enumerate(self.tokenized):
            tf_map = Counter(doc)
            dl = len(doc)
            s  = 0.0
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
    model   = EMBED_MODEL_DOC if doc_mode else EMBED_MODEL_QUERY
    payload = {"modelUri": model, "text": text[:2000]}
    for attempt in range(3):
        try:
            async with session.post(EMBED_URL, json=payload, headers=yandex_headers()) as r:
                if r.status == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return (await r.json())["embedding"]
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(1)


# ── [6] Сборка и загрузка индекса (numpy вместо pickle) ──────────────────────
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
            batch   = chunks[i:i + batch_size]
            tasks   = [embed_one(session, c["text"], doc_mode=True) for c in batch]
            results = await asyncio.gather(*tasks)
            embeddings.extend(results)
            done = min(i + batch_size, total)
            log.info(f"  {done}/{total} ({done * 100 // total}%)")
            await asyncio.sleep(0.6)

    emb_array = np.array(embeddings, dtype="float32")
    norms     = np.linalg.norm(emb_array, axis=1, keepdims=True)
    emb_array /= np.maximum(norms, 1e-8)

    np.save(INDEX_NPY_FILE, emb_array)
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    log.info(f"✅ Индекс сохранён: {INDEX_NPY_FILE} ({total} векторов)")


def load_index() -> tuple[np.ndarray, list[dict]]:
    # Новый numpy-формат
    if os.path.exists(INDEX_NPY_FILE) and os.path.exists(INDEX_META_FILE):
        emb = np.load(INDEX_NPY_FILE)
        with open(INDEX_META_FILE, encoding="utf-8") as f:
            chunks = json.load(f)
        log.info(f"Индекс загружен (numpy): {len(chunks)} векторов")
        return emb, chunks

    # Обратная совместимость: старый pickle → конвертируем и сохраняем
    if os.path.exists(INDEX_PKL_FILE):
        import pickle
        with open(INDEX_PKL_FILE, "rb") as f:
            data = pickle.load(f)
        log.info(f"Индекс загружен (pickle legacy): {len(data['chunks'])} — конвертирую в numpy...")
        np.save(INDEX_NPY_FILE, data["embeddings"])
        with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
            json.dump(data["chunks"], f, ensure_ascii=False)
        log.info("Конвертация завершена.")
        return data["embeddings"], data["chunks"]

    raise FileNotFoundError("Векторный индекс не найден. Запустите: python bot.py --build-index")


# ── [12] Hybrid retrieval: семантика + BM25 без hard-cut ─────────────────────
async def retrieve_hybrid(
    query: str,
    embeddings,
    chunks: list[dict],
    bm25: BM25,
    use_semantic: bool,
) -> tuple[list[dict], str]:
    """
    Если семантика дала score >= MIN_SCORE_SEM — используем её.
    Иначе — BM25 без порога отсечения (всегда что-то находим).
    Возвращает (результаты, метод).
    """
    sem_results = []

    if use_semantic and embeddings is not None:
        try:
            async with aiohttp.ClientSession() as session:
                q_emb = await embed_one(session, query, doc_mode=False)
            q_vec  = np.array(q_emb, dtype="float32")
            q_vec /= max(np.linalg.norm(q_vec), 1e-8)
            scores  = embeddings @ q_vec
            top_idx = np.argsort(scores)[::-1][:TOP_K]
            for idx in top_idx:
                c          = dict(chunks[idx])
                c["score"] = float(scores[idx])
                sem_results.append(c)
        except Exception as e:
            log.warning(f"Семантический поиск упал: {e} — переключаюсь на BM25")

    # Если семантика хорошая — возвращаем её
    if sem_results and sem_results[0]["score"] >= MIN_SCORE_SEM:
        return sem_results, "семантический"

    # [12] BM25 без hard-cut — нормализуем score и возвращаем лучшее
    bm25_raw = bm25.search(query, top_k=TOP_K)
    if bm25_raw:
        max_score = bm25_raw[0][1] if bm25_raw[0][1] > 0 else 1
        bm25_results = []
        for idx, score in bm25_raw:
            c          = dict(chunks[idx])
            c["score"] = score / max_score
            bm25_results.append(c)
        return bm25_results, "ключевой (BM25)"

    # Если вообще ничего не нашли — возвращаем слабые семантические результаты
    return sem_results, "семантический (низкий score)"


def build_context(relevant: list[dict]) -> str:
    parts, total = [], 0
    for c in relevant:
        block = f"[{c['title']}]\n{c['text']}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


# ── [18] DeepSeek + YandexGPT fallback ───────────────────────────────────────
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
        "max_tokens": 700,
    }
    async with session.post(
        DEEPSEEK_URL, json=payload, headers=headers,
        timeout=aiohttp.ClientTimeout(total=30)
    ) as r:
        r.raise_for_status()
        return (await r.json())["choices"][0]["message"]["content"]


async def yandexgpt_answer(session: aiohttp.ClientSession, system: str, user: str) -> str:
    payload = {
        "modelUri": YANDEX_GPT_MODEL,
        "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": 700},
        "messages": [
            {"role": "system", "text": system},
            {"role": "user",   "text": user},
        ],
    }
    async with session.post(
        YANDEX_GPT_URL, json=payload, headers=yandex_headers(),
        timeout=aiohttp.ClientTimeout(total=30)
    ) as r:
        r.raise_for_status()
        data = await r.json()
        return data["result"]["alternatives"][0]["message"]["text"]


async def get_answer(system: str, user: str) -> tuple[str, str]:
    """Пробует DeepSeek, при ошибке переключается на YandexGPT."""
    async with aiohttp.ClientSession() as session:
        try:
            answer = await deepseek_answer(session, system, user)
            return answer, "DeepSeek"
        except Exception as e:
            log.warning(f"DeepSeek недоступен ({e}), переключаюсь на YandexGPT...")
            try:
                answer = await yandexgpt_answer(session, system, user)
                return answer, "YandexGPT"
            except Exception as e2:
                log.error(f"YandexGPT тоже недоступен: {e2}")
                raise


# ── [10] Контекст диалога ─────────────────────────────────────────────────────
_MAX_DIALOGS = 1000  # [B] максимум чатов в памяти


def get_dialog_history(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> list[dict]:
    dialogs = context.bot_data.setdefault("dialogs", OrderedDict())
    return dialogs.get(str(chat_id), [])


def add_to_dialog(context: ContextTypes.DEFAULT_TYPE, chat_id: int, role: str, text: str):
    # [B] OrderedDict как LRU: при доступе перемещаем в конец
    dialogs: OrderedDict = context.bot_data.setdefault("dialogs", OrderedDict())
    key = str(chat_id)

    if key in dialogs:
        dialogs.move_to_end(key)  # освежаем — это активный чат
    else:
        dialogs[key] = []
        # [B] Если переполнили — выбрасываем самый старый чат
        if len(dialogs) > _MAX_DIALOGS:
            dialogs.popitem(last=False)

    dialogs[key].append({"role": role, "text": text[:300]})
    dialogs[key] = dialogs[key][-(DIALOG_HISTORY_LEN * 2):]


def build_prompt_with_history(query: str, context_text: str, history: list[dict]) -> str:
    parts = []
    if history:
        parts.append("История диалога:")
        for h in history:
            prefix = "Пользователь" if h["role"] == "user" else "Ассистент"
            parts.append(f"{prefix}: {h['text']}")
        parts.append("")
    parts.append(f"Текущий вопрос: {query}")
    parts.append(f"\nФрагменты нормативных актов:\n{context_text}")
    return "\n".join(parts)


# ── Анимация ──────────────────────────────────────────────────────────────────
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


# ── [4] Проверка админа по user_id ───────────────────────────────────────────
def is_admin(update: Update) -> bool:
    return update.effective_user.id == ADMIN_USER_ID


# ── [13] Разбивка длинных сообщений ──────────────────────────────────────────
async def send_long_message(target, text: str, parse_mode: str = "Markdown"):
    MAX = 4000
    if len(text) <= MAX:
        await target.edit_text(text, parse_mode=parse_mode)
        return

    parts = []
    while text:
        if len(text) <= MAX:
            parts.append(text)
            break
        split_at = text.rfind("\n", 0, MAX)
        if split_at == -1:
            split_at = MAX
        parts.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    await target.edit_text(parts[0], parse_mode=parse_mode)
    for part in parts[1:]:
        await target.reply_text(part, parse_mode=parse_mode)


# ── Команды пользователя ──────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Привет!* Я бот по транспортному законодательству РФ.\n\n"
        "📚 *База знаний (21 документ):*\n"
        "• ПДД РФ 2026 (полный текст)\n"
        "• КоАП — нарушения ПДД, транспорт, таможня\n"
        "• УАТ — Устав автомобильного транспорта\n"
        "• Правила перевозки грузов и тяжеловесов\n"
        "• ЭТРН, ФЗ-87, ГК, УК, НК — транспортные разделы\n"
        "• Нормы рабочего времени водителей\n"
        "• Судебная практика ВС РФ по перевозкам\n\n"
        "💬 *Примеры вопросов:*\n"
        "_— Штраф за превышение на 40 км/ч?_\n"
        "_— Сколько часов водитель может ехать без отдыха?_\n"
        "_— Ответственность экспедитора за утрату груза?_",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ *Как пользоваться*\n\n"
        "Задайте вопрос — получите краткий ответ со ссылкой на статью.\n"
        "Можно задавать уточняющие вопросы — бот помнит контекст диалога.\n\n"
        "*Примеры:*\n"
        "• Штраф за езду без ОСАГО?\n"
        "• Норма непрерывного вождения?\n"
        "• Когда нужна ТТН, а когда ЭТРН?\n"
        "• НДС 0% при международных перевозках?\n"
        "• Срок давности по договору экспедиции?\n\n"
        "⚠️ Бот даёт справочную информацию. По конкретным ситуациям рекомендуем консультацию юриста.",
        parse_mode="Markdown",
    )


# ── Основной обработчик ───────────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user  = update.effective_user
    query = update.message.text.strip()
    if not query:
        return

    # [5] Ограничение длины
    if len(query) > MAX_QUERY_LEN:
        await update.message.reply_text(
            f"⚠️ Слишком длинный запрос (максимум {MAX_QUERY_LEN} символов). "
            "Пожалуйста, сформулируйте вопрос короче."
        )
        return

    # Проверка блокировки
    if is_blocked(user.id):
        await update.message.reply_text("🚫 Вы заблокированы.")
        return

    # [3] Rate limiting
    wait = check_rate_limit(user.id)
    if wait > 0:
        await update.message.reply_text(
            f"⏱ Подождите ещё {int(wait) + 1} сек. перед следующим вопросом."
        )
        return

    # Статистика — в фоне, не тормозим ответ
    # [C] Имя таска для отслеживания в post_stop
    task = asyncio.create_task(
        record_question(user.id, user.username or str(user.id), query)
    )
    task.set_name(f"record_question_{user.id}")

    # Анимация
    thinking_msg = await update.message.reply_text("⏳ Ищу в базе законов...")
    stop_event   = asyncio.Event()
    anim_task    = asyncio.create_task(animate_thinking(thinking_msg, stop_event))

    chunks       = context.bot_data["chunks"]
    bm25         = context.bot_data["bm25"]
    use_semantic = context.bot_data.get("use_semantic", False)
    embeddings   = context.bot_data.get("embeddings")

    try:
        # [12] Hybrid search
        relevant, search_method = await retrieve_hybrid(
            query, embeddings, chunks, bm25, use_semantic
        )

        if not relevant:
            stop_event.set()
            await anim_task
            await thinking_msg.edit_text(
                "🤔 Не нашёл информации по этому вопросу в базе законов.\n"
                "Попробуйте переформулировать."
            )
            return

        ctx = build_context(relevant)

        # [10] Контекст диалога
        history  = get_dialog_history(context, update.effective_chat.id)
        user_msg = build_prompt_with_history(query, ctx, history)

        # [18] DeepSeek с fallback на YandexGPT
        answer, llm_name = await get_answer(SYSTEM_PROMPT, user_msg)

        # Сохраняем в историю диалога [10]
        add_to_dialog(context, update.effective_chat.id, "user", query)
        add_to_dialog(context, update.effective_chat.id, "assistant", answer)

        # Источники
        seen, sources = set(), []
        for c in relevant[:4]:
            if c["title"] not in seen:
                seen.add(c["title"])
                sources.append(c["title"])

        footer = f"\n\n📌 *Источники* (🔍 {search_method}):\n" + "\n".join(f"• {s}" for s in sources)
        full   = answer + footer

        stop_event.set()
        await anim_task

        # [13] Разбивка на части если ответ длинный
        await send_long_message(thinking_msg, full)

    except Exception as e:
        stop_event.set()
        await anim_task
        log.exception(f"Ошибка при обработке запроса от {user.id}: {e}")
        await thinking_msg.edit_text(
            "⚠️ Сервис временно недоступен. Попробуйте через минуту."
        )


# ── Админ-команды ─────────────────────────────────────────────────────────────
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return
    stats   = load_stats()
    blocked = load_blocked()
    await update.message.reply_text(
        f"🔧 *Панель администратора*\n\n"
        f"👥 Пользователей: {len(stats['users'])}\n"
        f"❓ Вопросов всего: {stats.get('total_questions', 0)}\n"
        f"🚫 Заблокировано: {len(blocked)}\n"
        f"🔍 Поиск: {'семантический ✅' if context.bot_data.get('use_semantic') else 'BM25'}\n\n"
        f"*Команды:*\n"
        f"/stats — статистика топ-10\n"
        f"/user @username — инфо о пользователе\n"
        f"/block @username — заблокировать\n"
        f"/unblock @username — разблокировать\n"
        f"/blocked — список заблокированных",
        parse_mode="Markdown",
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return
    stats = load_stats()
    users = stats.get("users", {})
    top   = sorted(users.items(), key=lambda x: x[1].get("questions", 0), reverse=True)[:10]
    lines = [
        "📊 *Статистика бота*\n",
        f"Всего вопросов: {stats.get('total_questions', 0)}",
        f"Всего пользователей: {len(users)}\n",
        "*Топ-10 активных:*",
    ]
    for uid, u in top:
        lines.append(
            f"• @{u.get('username', uid)}: {u.get('questions', 0)} вопр. "
            f"(посл: {u.get('last_seen', '—')})"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text("⛔ Нет доступа.")
        return
    if not context.args:
        await update.message.reply_text("Использование: /user @username или /user ID")
        return
    target = context.args[0].lstrip("@").lower()
    stats  = load_stats()
    found  = None
    for uid, u in stats.get("users", {}).items():
        if (u.get("username") or "").lower() == target or uid == target:
            found = (uid, u)
            break
    if not found:
        await update.message.reply_text(f"Пользователь {target} не найден.")
        return
    uid, u = found
    history_text = "\n".join(
        f"  {h['t']}: {h['q']}" for h in u.get("history", [])[-10:]
    ) or "  нет"
    await update.message.reply_text(
        f"👤 *@{u.get('username', uid)}*\n"
        f"ID: `{uid}`\n"
        f"Вопросов: {u.get('questions', 0)}\n"
        f"Последний визит: {u.get('last_seen', '—')}\n\n"
        f"*Последние вопросы:*\n{history_text}",
        parse_mode="Markdown",
    )


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
    found_id = None
    for uid, u in stats.get("users", {}).items():
        if (u.get("username") or "").lower() == target:
            found_id = int(uid)
            break
    if not found_id:
        try:
            found_id = int(target)
        except ValueError:
            await update.message.reply_text(f"Пользователь {target} не найден в базе.")
            return
    if found_id in blocked:
        await update.message.reply_text("Пользователь уже заблокирован.")
        return
    blocked.append(found_id)
    await save_blocked(blocked)
    await update.message.reply_text(f"🚫 Заблокирован: {target} (ID: {found_id}).")


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
            await update.message.reply_text(f"Пользователь {target} не найден.")
            return
    if found_id not in blocked:
        await update.message.reply_text("Пользователь не заблокирован.")
        return
    blocked.remove(found_id)
    await save_blocked(blocked)
    await update.message.reply_text(f"✅ Разблокирован: {target}.")


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
        lines.append(f"• @{u.get('username', str(uid))} (ID: {uid})")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ── Startup / Shutdown ────────────────────────────────────────────────────────
async def post_init(application):
    log.info("Загружаю базу знаний...")
    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)
    chunks = kb["chunks"]

    log.info(f"Строю BM25 индекс ({len(chunks)} чанков)...")
    bm25 = BM25([c["text"] for c in chunks])
    application.bot_data["chunks"] = chunks
    application.bot_data["bm25"]   = bm25

    # [6] Загружаем numpy-индекс (или конвертируем из pickle)
    try:
        embeddings, _ = load_index()
        application.bot_data["embeddings"]   = embeddings
        application.bot_data["use_semantic"] = True
        log.info("✅ Семантический поиск активен (Yandex Embeddings)")
    except FileNotFoundError:
        log.info("ℹ️ Векторный индекс не найден — только BM25. Запустите --build-index")
        application.bot_data["use_semantic"] = False
    except Exception as e:
        log.warning(f"Не удалось загрузить индекс: {e} — используется BM25")
        application.bot_data["use_semantic"] = False

    await application.bot.set_my_commands([
        BotCommand("start",   "О боте и примеры вопросов"),
        BotCommand("help",    "Как пользоваться"),
        BotCommand("admin",   "Панель администратора"),
        BotCommand("stats",   "Статистика (только админ)"),
        BotCommand("user",    "Инфо о пользователе (только админ)"),
        BotCommand("block",   "Заблокировать (только админ)"),
        BotCommand("unblock", "Разблокировать (только админ)"),
        BotCommand("blocked", "Список заблокированных (только админ)"),
    ])
    log.info("✅ Бот готов к работе")


async def post_stop(application):
    # [C] Дожидаемся незавершённых record_question перед выходом
    pending = [
        t for t in asyncio.all_tasks()
        if t.get_name().startswith("record_question") and not t.done()
    ]
    if pending:
        log.info(f"Завершаю {len(pending)} незаписанных stat-тасков...")
        await asyncio.gather(*pending, return_exceptions=True)
    log.info("Бот остановлен.")


def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_stop(post_stop)
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
    parser.add_argument("--build-index", action="store_true", help="Пересобрать векторный индекс")
    args = parser.parse_args()
    if args.build_index:
        asyncio.run(build_index())
    else:
        main()
