"""
D&D Core AI — гибридная архитектура B.

Pipeline хода (после оптимизации):
    Игрок → heuristic_classify (Python, без LLM)
          → LogicAgent.parse_intent (Qwen 1.5B + LoRA) ← чарник игрока
          → DnDEngine (Python, кубики и DC)
          → TacticalAI (Python, behavior_tag)
          → NarrativeAgent (Qwen 7B Q4 через Ollama) ← чарник + история
          → LogicAgent.state_delta → world_state.

ИЗМЕНЕНИЯ ПО СРАВНЕНИЮ С ПРЕДЫДУЩЕЙ ВЕРСИЕЙ:
    1. LLM-classify заменён на чистый Python: экономит ~3-5 сек на ход.
    2. Загружается character_sheet.json — игроку даны статы, навыки, оружие.
       Judge и Narrative видят чарник.
    3. Логи дублируются в training/logs/dnd_ai.log (с rotating).
    4. Запрос информации («вспомнить», «что я вижу») обрабатывается специально —
       не превращается в движение.
"""

import os
import re
import sys
import json
import socket
import time
import gc
import asyncio
import logging
import warnings
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from typing import Optional

# Подавляем известные информационные warning'и которые не влияют на работу.
# 1) peft 0.17 ругается на неизвестные поля в adapter_config.json от свежего peft 0.19 (Colab).
#    Они все опциональные и игнорируются — это нормально, см. README.
warnings.filterwarnings(
    "ignore",
    message=r"Unexpected keyword arguments.*for class LoraConfig.*",
    category=UserWarning,
)

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dnd_engine import DnDEngine
from tactical_ai import TacticalAI
from logic_agent import LogicAgent
from narrative_agent import NarrativeAgent
from npc_registry import (
    NPC_TEMPLATES, NPC_ALIASES,
    canonicalize_target, ensure_npc_spawned, name_for_entity,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
SESSIONS_DIR = os.path.join(LOGS_DIR, "sessions")
ADAPTERS_BASE_PATH = os.path.join(PROJECT_ROOT, "dnd_core_models")
CHARSHEET_PATH = os.path.join(ADAPTERS_BASE_PATH, "character_sheet.json")


# === LOGGING ===============================================================

# Каждый запуск сервера = отдельная папка с логами.
# Это удобно для отладки: видишь как изменилось поведение между правками кода,
# можно сравнивать сессии бок-о-бок.

os.makedirs(SESSIONS_DIR, exist_ok=True)

SESSION_ID = time.strftime("%Y-%m-%d_%H-%M-%S")
SESSION_DIR = os.path.join(SESSIONS_DIR, f"session_{SESSION_ID}")
os.makedirs(SESSION_DIR, exist_ok=True)

SESSION_LOG_PATH = os.path.join(SESSION_DIR, "dnd_ai.log")
SESSION_TURNS_PATH = os.path.join(SESSION_DIR, "turns.jsonl")

# Указатель «вот последняя сессия» — для быстрого доступа без копания в timestamp-папках
with open(os.path.join(LOGS_DIR, "latest_session.txt"), "w", encoding="utf-8") as f:
    f.write(SESSION_DIR + "\n")

# Чистим старые сессии — оставляем 10 последних, чтобы папка logs не разрасталась
def _cleanup_old_sessions(keep: int = 10):
    try:
        all_sessions = sorted(
            [d for d in os.listdir(SESSIONS_DIR) if d.startswith("session_")],
            reverse=True,
        )
        for old in all_sessions[keep:]:
            old_path = os.path.join(SESSIONS_DIR, old)
            try:
                import shutil
                shutil.rmtree(old_path)
            except Exception:
                pass
    except Exception:
        pass


_cleanup_old_sessions(keep=10)


# Atomic file handler: open → write → close на каждую запись.
# Стандартный FileHandler держит файл открытым весь жизненный цикл процесса.
# На OneDrive / Dropbox / Docker bind mounts / антивирус EDR такой handle
# часто перехватывается синк-агентами, и записи перестают доходить до диска.
# Atomic-вариант медленнее (открытие файла стоит ~1ms на запись), но это
# не важно для лога 5-10 строк на ход. Гарантия записи важнее скорости.
class AtomicFileHandler(logging.Handler):
    def __init__(self, path: str, mode: str = "a", encoding: str = "utf-8"):
        super().__init__()
        self.path = path
        self.encoding = encoding
        # Если запуск новой сессии — обнуляем файл.
        # На followup записях используем append.
        if mode == "w":
            try:
                with open(path, "w", encoding=encoding):
                    pass
            except Exception as e:
                print(f"[atomic-handler] не могу создать {path}: {e}", flush=True)

    def emit(self, record):
        try:
            msg = self.format(record)
            with open(self.path, "a", encoding=self.encoding) as f:
                f.write(msg + "\n")
        except Exception:
            self.handleError(record)


# Главный логгер всего сервера — пишет в файл текущей сессии + консоль
logger = logging.getLogger("dnd")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.propagate = False

_file_h = AtomicFileHandler(SESSION_LOG_PATH, mode="w")
_file_h.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_file_h.setLevel(logging.DEBUG)
logger.addHandler(_file_h)

_console_h = logging.StreamHandler()
_console_h.setFormatter(logging.Formatter("%(message)s"))
_console_h.setLevel(logging.INFO)
logger.addHandler(_console_h)

# Отдельный JSON-логгер для машинного парсинга ходов
turn_logger = logging.getLogger("dnd.turn")
turn_logger.setLevel(logging.INFO)
turn_logger.handlers.clear()
turn_logger.propagate = False
_turn_h = AtomicFileHandler(SESSION_TURNS_PATH, mode="w")
_turn_h.setFormatter(logging.Formatter("%(message)s"))
turn_logger.addHandler(_turn_h)

# Самый ранний sanity-чек — пишем сразу после настройки handler'ов.
logger.info(f"[boot] Логгер инициализирован. Сессия: session_{SESSION_ID}")
print(f"[boot] >>> stdout: logger initialized, session_{SESSION_ID}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Современная замена @app.on_event('startup'). Прогрев агентов при старте сервера."""
    print("[lifespan] >>> startup begin", flush=True)
    logger.info("=" * 60)
    logger.info("[lifespan] startup begin")
    logger.info(f"[i] Device: {DEVICE}  |  Torch: {torch.__version__}")
    logger.info(f"[i] Quantization (bitsandbytes 4-bit): {'ВКЛ' if QUANT_AVAILABLE else 'ВЫКЛ'}")
    logger.info(f"[i] Архитектура: гибрид B (Logic 1.5B + Narrative 7B + Python-тактик)")
    logger.info(f"[i] Сессия: session_{SESSION_ID}")
    logger.info(f"[i] Логи этой сессии: {SESSION_LOG_PATH}")
    logger.info(f"[i] JSON-логи ходов:  {SESSION_TURNS_PATH}")
    logger.info(f"[i] Указатель на последнюю сессию: {os.path.join(LOGS_DIR, 'latest_session.txt')}")

    try:
        LOGIC.load()
        logger.info("[+] Logic-агент прогрет")
    except Exception as e:
        logger.error(f"[!] Logic не загрузился: {e}")

    ok, msg = NARRATIVE.verify()
    if ok:
        logger.info(f"[+] Narrative-агент (Ollama): {msg}")
    else:
        logger.warning(f"[!] Narrative-агент НЕ ГОТОВ: {msg}")

    logger.info(f"[+] Загружен чарник: {CHARACTER.get('name')} (HP {CHARACTER.get('hp_current')}/{CHARACTER.get('hp_max')})")
    logger.info("=" * 60)
    yield
    # На shutdown ничего особенного не делаем — Python приберёт сам


app = FastAPI(title="D&D Core AI — Hybrid B", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=4)

PYTHON_SERVER_HOST = os.environ.get("PYTHON_SERVER_HOST", "0.0.0.0")
PYTHON_SERVER_PORT = int(os.environ.get("PYTHON_SERVER_PORT", 8000))


# === DEVICE DETECTION =====================================================

def has_directml() -> bool:
    try:
        import torch_directml  # noqa
        return True
    except Exception:
        return False


def has_rocm() -> bool:
    try:
        return getattr(torch.version, "hip", None) is not None
    except Exception:
        return False


def get_device():
    if torch.cuda.is_available() and not has_rocm():
        return "cuda"
    if has_rocm():
        return "rocm"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if has_directml():
        return "dml"
    return "cpu"


def get_torch_device(name: str):
    if name in ("cuda", "rocm"):
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "dml":
        import torch_directml
        return torch_directml.device()
    return torch.device("cpu")


DEVICE = get_device()
TORCH_DEVICE = get_torch_device(DEVICE)
QUANT_AVAILABLE = (DEVICE == "cuda")


# === CHARACTER SHEET =======================================================

def load_character_sheet() -> dict:
    """Загружает чарник игрока. Если файла нет — возвращает минимум."""
    if os.path.exists(CHARSHEET_PATH):
        with open(CHARSHEET_PATH, encoding="utf-8") as f:
            sheet = json.load(f)
        logger.info(f"[char] Загружен чарник: {sheet.get('name')} ({sheet.get('class')} ур.{sheet.get('level')})")
        return sheet
    logger.warning("[char] character_sheet.json не найден — используется заглушка")
    return {
        "name": "Безымянный", "class": "наёмник", "level": 1,
        "hp_current": 20, "hp_max": 20, "ac": 13,
        "ability_modifiers": {"str": 2, "dex": 2, "con": 1, "int": 0, "wis": 1, "cha": 0},
        "proficiency_bonus": 2,
        "weapons": [], "skills": {}, "spells": [],
    }


CHARACTER = load_character_sheet()


def character_brief() -> str:
    """Краткое описание чарника для промпта Narrative (укладывается в ~150 токенов).

    ВАЖНО: имя и класс НЕ включаются — Narrative не должен использовать имя
    в третьем лице. Только оружие/заклинания/общие черты как СПРАВКА для нарратива.
    """
    weapons = CHARACTER.get("weapons", [])
    spells = CHARACTER.get("spells", [])

    lines = ["У игрока есть (для деталей описания):"]
    if weapons:
        wpn_names = [w.get('name', '') for w in weapons]
        lines.append(f"- Оружие: {', '.join(wpn_names)}")
    if spells:
        spell_names = [s.get('name', '') for s in spells]
        lines.append(f"- Известные заклинания: {', '.join(spell_names)}")
    lines.append(f"- HP сейчас: {CHARACTER.get('hp_current',0)}/{CHARACTER.get('hp_max',0)}")
    return "\n".join(lines)


def get_skill_modifier(skill_name: str) -> int:
    """Возвращает суммарный бонус для проверки навыка (ability + proficiency)."""
    skill = CHARACTER.get("skills", {}).get(skill_name.lower())
    if skill:
        return int(skill.get("bonus", 0))
    return 0


def get_weapon_data(weapon_id: str) -> dict:
    for w in CHARACTER.get("weapons", []):
        if w.get("id") == weapon_id or w.get("name") == weapon_id:
            return w
    return {}


# === WHITELIST из чарника — что персонаж РЕАЛЬНО умеет ===

def build_allowed_lists():
    """Источник истины — character_sheet.json. Все списки динамические."""
    weapons = {w["id"]: w["name"] for w in CHARACTER.get("weapons", [])}
    spells = {s["id"]: s.get("name", s["id"]) for s in CHARACTER.get("spells", [])}
    skills = list(CHARACTER.get("skills", {}).keys())
    return weapons, spells, skills


ALLOWED_WEAPONS, ALLOWED_SPELLS, ALLOWED_SKILLS = build_allowed_lists()


def build_judge_system_prompt() -> str:
    """Динамический system prompt с whitelist'ом из чарника.

    Принцип: даём модели ЯВНЫЙ список того, что персонаж умеет.
    Всё остальное → action='impossible'. Это надёжнее чем перечислять что нельзя
    (всегда можно придумать новый запрещённый предмет).
    """
    weapons_lines = [f"  - {wid}: {wname}" for wid, wname in ALLOWED_WEAPONS.items()] or ["  (нет оружия)"]
    spells_lines = [f"  - {sid}: {sname}" for sid, sname in ALLOWED_SPELLS.items()] or ["  (нет заклинаний)"]
    skills_line = ", ".join(ALLOWED_SKILLS) if ALLOWED_SKILLS else "(нет тренированных)"

    return (
        "Ты — Судья D&D 5e. Твоя ЕДИНСТВЕННАЯ задача: распарсить русскую фразу игрока в строгий JSON.\n"
        "НЕ принимай решений об успехе/провале — это сделает движок. НЕ катай кубики.\n"
        "\n"
        "ПЕРСОНАЖ ИГРОКА МОЖЕТ ИСПОЛЬЗОВАТЬ ТОЛЬКО следующее (whitelist):\n"
        "\n"
        "ОРУЖИЕ (используй ровно эти id в поле 'weapon'):\n"
        + "\n".join(weapons_lines) +
        "\n\n"
        "ЗАКЛИНАНИЯ (используй ровно эти id в поле 'spell'):\n"
        + "\n".join(spells_lines) +
        "\n\n"
        f"НАВЫКИ (используй один из этих id в поле 'skill'): {skills_line}\n"
        "\n"
        "ПРАВИЛО: если игрок пытается использовать ОРУЖИЕ или ЗАКЛИНАНИЕ, "
        "которого нет в whitelist выше — action='impossible' и заполни 'reason'. "
        "Не подменяй на похожее, не выдумывай новые id.\n"
        "\n"
        "Также невозможны: полёт без крыльев и магии, телепортация, "
        "остановка времени, чтение мыслей, оживление мёртвых без божественной магии.\n"
        "\n"
        "Верни ровно один JSON и ничего больше:\n"
        '{"action":"attack"|"skill_check"|"spell"|"dialogue"|"exploration"|"impossible",'
        '"target":<строка или null>,"skill":<id навыка или null>,"weapon":<id оружия из whitelist или null>,'
        '"spell":<id заклинания из whitelist или null>,"modifier":<целое>,'
        '"advantage":<bool>,"disadvantage":<bool>,'
        '"difficulty":"easy"|"medium"|"hard"|null,'
        '"damage_dice":<строка типа "1d8+3" или null>,"reason":<строка>}'
    )


# === AGENTS ===============================================================

LOGIC_ADAPTER_PATH = os.path.join(ADAPTERS_BASE_PATH, "logic_lora_adapter")
LOGIC = LogicAgent(
    base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path=LOGIC_ADAPTER_PATH if os.path.exists(os.path.join(LOGIC_ADAPTER_PATH, "adapter_config.json")) else None,
    device=DEVICE,
    torch_device=TORCH_DEVICE,
    quant_available=QUANT_AVAILABLE,
)
NARRATIVE = NarrativeAgent()


# === PENCIL FILTER + ANTI-GOD-MODDING =====================================

_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)
_META_RE = re.compile(r"---+\s*\n?\s*\(?Примечание автора.*?$", re.DOTALL | re.IGNORECASE)

_GODMOD_PATTERNS = [
    # Эмоции / решения — игрок сам должен решить как реагировать
    (re.compile(r"\bвы\s+(?:решаете|думаете|чувствуете|боитесь|пугаетесь|радуетесь|"
                r"хотите|желаете|надеетесь|сомневаетесь|не хотите|не желаете)\b", re.IGNORECASE),
     "перед вами"),
    # Действия — игрок сам должен выбрать движение
    (re.compile(r"\bвы\s+(?:бежите|убегаете|нападаете|сдаётесь|соглашаетесь|отказываетесь|"
                r"идёте|идете|уходите|поворачиваетесь|возвращаетесь|останавливаетесь)\b", re.IGNORECASE),
     "перед вами выбор"),
    # Интерпретация мысли / понимания — игрок сам решает что подумал
    (re.compile(r"\bвы\s+(?:понимаете|осознаёте|осознаете|догадываетесь|подозреваете|"
                r"предполагаете|приходите к выводу|делаете вывод)\b", re.IGNORECASE),
     "перед вами становится ясно"),
    # Память / знание — игрок сам решает что вспомнил
    (re.compile(r"\bвы\s+(?:знаете|помните|помнили|узнаёте|узнаете|"
                r"вспоминаете о|припоминаете)\b", re.IGNORECASE),
     "в памяти всплывает"),
    # Речь — игрок сам должен сказать что говорит
    (re.compile(r"\bвы\s+(?:говорите|шепчете|кричите|молвите|произносите|восклицаете|"
                r"обращаетесь к|спрашиваете)\b", re.IGNORECASE),
     "ваши губы готовы произнести"),
    # Мимика — игрок сам решает выражение лица
    (re.compile(r"\bвы\s+(?:улыба|хмурит|кивает|кивн|вздыха|плач|смеёт|смеет|"
                r"морщ|насмеха|подмигив)\w*\b", re.IGNORECASE),
     "ваше лицо реагирует"),
]

_ANACHRONISM_REPLACEMENTS = [
    (re.compile(r"\bтелефонн\w*\b", re.IGNORECASE),    "магических"),
    (re.compile(r"\bтелефон\w*\b", re.IGNORECASE),     "магическ"),
    (re.compile(r"\bпровод\w*\b", re.IGNORECASE),      "верёвок"),
    (re.compile(r"\bкабел\w*\b", re.IGNORECASE),       "верёвок"),
    (re.compile(r"\bэлектричеств\w*\b", re.IGNORECASE), "магии"),
    (re.compile(r"\bлампочк\w*\b", re.IGNORECASE),     "факел"),
    (re.compile(r"\bбатаре\w*\b", re.IGNORECASE),      "кристалл"),
    (re.compile(r"\bкомпьютер\w*\b", re.IGNORECASE),   "артефакт"),
    (re.compile(r"\bтелевизор\w*\b", re.IGNORECASE),   "хрустальный шар"),
    (re.compile(r"\bкалькулятор\w*\b", re.IGNORECASE), "казначей"),
    (re.compile(r"\bсамолёт\w*\b", re.IGNORECASE),     "птица"),
    (re.compile(r"\bсамолет\w*\b", re.IGNORECASE),     "птица"),
    (re.compile(r"\bракет\w*\b", re.IGNORECASE),       ""),  # ракета — отдельная история, удаляем
    (re.compile(r"\bтрансформер\w*\b", re.IGNORECASE), ""),
    (re.compile(r"\bnoir\b", re.IGNORECASE),           "мрачн"),
    (re.compile(r"\bfashion\w*\b", re.IGNORECASE),     "одеяний"),
    (re.compile(r"\bвиртуально\w*\b", re.IGNORECASE),  "будто"),
    (re.compile(r"\bоштукатуренн\w*\b", re.IGNORECASE),"каменн"),
    (re.compile(r"\bстеклянн\w*\b", re.IGNORECASE),    "каменн"),
]


# Английские слова которые Qwen иногда вставляет посреди русского текста.
# Заменяем на пустую строку (вырежется) или русский эквивалент.
# Это безопасно — это технические термины которые НЕ должны быть в нарративе.
_ENGLISH_LEAK_REPLACEMENTS = [
    # D&D механические термины (имена навыков)
    (re.compile(r"\bathletics\b", re.IGNORECASE),       "силе"),
    (re.compile(r"\bacrobatics\b", re.IGNORECASE),      "ловкости"),
    (re.compile(r"\bstealth\b", re.IGNORECASE),         "скрытности"),
    (re.compile(r"\bperception\b", re.IGNORECASE),      "внимательности"),
    (re.compile(r"\bsurvival\b", re.IGNORECASE),        "выживанию"),
    (re.compile(r"\binvestigation\b", re.IGNORECASE),   "анализу"),
    (re.compile(r"\binsight\b", re.IGNORECASE),         "интуиции"),
    (re.compile(r"\barcana\b", re.IGNORECASE),          "магии"),
    (re.compile(r"\bdeception\b", re.IGNORECASE),       "обману"),
    (re.compile(r"\bpersuasion\b", re.IGNORECASE),      "убеждению"),
    (re.compile(r"\bintimidation\b", re.IGNORECASE),    "запугиванию"),
    (re.compile(r"\bhistory\b", re.IGNORECASE),         "истории"),
    (re.compile(r"\bnature\b", re.IGNORECASE),          "природе"),
    # Базовые англ. слова которые модель иногда не переводит
    (re.compile(r"\bmetal\b", re.IGNORECASE),           "металл"),
    (re.compile(r"\bwood\b", re.IGNORECASE),            "дерево"),
    (re.compile(r"\bstone\b", re.IGNORECASE),           "камень"),
    (re.compile(r"\battack\b", re.IGNORECASE),          "атака"),
    (re.compile(r"\bdefense\b", re.IGNORECASE),         "защита"),
    (re.compile(r"\bdamage\b", re.IGNORECASE),          "урон"),
    (re.compile(r"\bhit\b", re.IGNORECASE),             "удар"),
    (re.compile(r"\bmiss\b", re.IGNORECASE),            "промах"),
    (re.compile(r"\broll\b", re.IGNORECASE),            "бросок"),
    (re.compile(r"\bcheck\b", re.IGNORECASE),           "проверка"),
    (re.compile(r"\bturn\b", re.IGNORECASE),            "ход"),
    (re.compile(r"\bweapon\b", re.IGNORECASE),          "оружие"),
    (re.compile(r"\barmor\b", re.IGNORECASE),           "доспех"),
    # Часто пролетающие модальности
    (re.compile(r"\bok\b", re.IGNORECASE),              "хорошо"),
    (re.compile(r"\byes\b", re.IGNORECASE),             "да"),
    (re.compile(r"\bno\b", re.IGNORECASE),              "нет"),
]


# Согласование глаголов после «вы»: модель копирует фразу игрока «я бью»
# и заменяет только «я» → «вы», получается уродливое «вы бью». Эта мапа
# исправляет частые формы 1-го лица на корректные 2-го лица множ.ч.
_FIRST_TO_SECOND_PERSON = [
    # Прямое спряжение часто употребляемых глаголов «я Х → ты делаешь → вы Хёте/Хите»
    (re.compile(r"\bвы\s+бью\b", re.IGNORECASE),         "вы бьёте"),
    (re.compile(r"\bвы\s+рублю\b", re.IGNORECASE),       "вы рубите"),
    (re.compile(r"\bвы\s+режу\b", re.IGNORECASE),        "вы режете"),
    (re.compile(r"\bвы\s+стреляю\b", re.IGNORECASE),     "вы стреляете"),
    (re.compile(r"\bвы\s+атакую\b", re.IGNORECASE),      "вы атакуете"),
    (re.compile(r"\bвы\s+убиваю\b", re.IGNORECASE),      "вы убиваете"),
    (re.compile(r"\bвы\s+бросаю\b", re.IGNORECASE),      "вы бросаете"),
    (re.compile(r"\bвы\s+кидаю\b", re.IGNORECASE),       "вы кидаете"),
    (re.compile(r"\bвы\s+иду\b", re.IGNORECASE),         "вы идёте"),
    (re.compile(r"\bвы\s+ищу\b", re.IGNORECASE),         "вы ищете"),
    (re.compile(r"\bвы\s+вижу\b", re.IGNORECASE),        "вы видите"),
    (re.compile(r"\bвы\s+слышу\b", re.IGNORECASE),       "вы слышите"),
    (re.compile(r"\bвы\s+касту\w*\b", re.IGNORECASE),    "вы кастуете"),
    (re.compile(r"\bвы\s+произнесу\b", re.IGNORECASE),   "вы произносите"),
]


def _build_person_swap_patterns():
    """Динамические regex'ы: имя персонажа (и его части) в третьем лице → «вы».

    Защита от «Карран Тёмный поднимает меч» когда должно быть «вы поднимаете меч».
    Маленькая модель часто срывается в третье лицо когда видит имя в промпте.
    """
    name = CHARACTER.get("name", "")
    if not name:
        return []
    # Разбиваем «Карран Тёмный» на ['Карран', 'Тёмный'] — каждое имя по отдельности
    parts = [p for p in name.split() if len(p) > 2]
    patterns = []
    # «Карран Тёмный» (полное имя) → «вы»
    patterns.append((re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE), "вы"))
    # Каждое слово имени отдельно (Карран, Тёмный)
    for part in parts:
        patterns.append((re.compile(rf"\b{re.escape(part)}\b", re.IGNORECASE), "вы"))
    # Класс и роль персонажа в третьем лице
    cls = CHARACTER.get("class", "")
    if cls:
        patterns.append((re.compile(rf"\b(?:следопыт|разведчик|охотник)\w*\b", re.IGNORECASE), "вы"))
    return patterns


_PERSON_SWAP_PATTERNS = _build_person_swap_patterns()


# Whitelist-подход: оставляем ТОЛЬКО то что может быть в нормальном русском тексте.
# Всё остальное (иероглифы CJK, китайская пунктуация ，。, японские каны,
# иврит, арабский, корейский, эмодзи, тайский) — вырезается.
#
# Главный фикс: ранее ловили только CJK Unified Ideographs (一-鿿).
# CJK-пунктуация (　-〿) и Fullwidth-формы (＀-￯) проходили —
# из-за этого ， и 。 в нарративе попадали в history, и модель уходила
# в петлю китайского. Теперь они тоже вырезаются.
_NON_RUSSIAN_SCRIPTS = re.compile(
    "["
    "一-鿿"          # CJK Unified Ideographs
    "㐀-䶿"          # CJK Extension A
    "　-〿"   # CJK Symbols and Punctuation (，。「」『』 и т.д.)
    "＀-￯"   # Halfwidth/Fullwidth forms (，。１２ и т.д.)
    "぀-ゟ"          # Hiragana
    "゠-ヿ"          # Katakana
    "가-힯"          # Hangul (корейский)
    "֐-׿"            # Hebrew
    "؀-ۿ"            # Arabic
    "฀-๿"            # Thai
    "✀-➿"            # Dingbats / эмодзи
    "\U0001F000-\U0001FFFF"  # Эмодзи и символы
    "]+",
    re.UNICODE,
)


def pencil_filter(text: str) -> str:
    text = _THOUGHT_RE.sub("", text)
    text = _META_RE.sub("", text)
    # Удаляем не-русские письменности (кит/яп/иврит/араб/эмодзи + CJK-пунктуация)
    text = _NON_RUSSIAN_SCRIPTS.sub("", text)
    # Заменяем английские слова-мусоры (athletics, metal, attack...) на русские эквиваленты
    for pat, replacement in _ENGLISH_LEAK_REPLACEMENTS:
        text = pat.sub(replacement, text)
    for pat, replacement in _GODMOD_PATTERNS:
        text = pat.sub(replacement, text)
    for pat, replacement in _ANACHRONISM_REPLACEMENTS:
        text = pat.sub(replacement, text)
    # Имя персонажа в третьем лице → «вы»
    for pat, replacement in _PERSON_SWAP_PATTERNS:
        text = pat.sub(replacement, text)
    # «вы бью» → «вы бьёте» (согласование лица)
    for pat, replacement in _FIRST_TO_SECOND_PERSON:
        text = pat.sub(replacement, text)
    # «вы вы» (от двойной замены имени) → «вы»
    text = re.sub(r"\bвы\s+вы\b", "вы", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip()


# === HEURISTIC CLASSIFIER (без LLM) =======================================

# Минимальный blacklist — только то что нарушает ФИЗИКУ мира (а не whitelist чарника).
# Современное оружие/техника отсекается через whitelist оружий в чарнике —
# если базуки нет в weapons[], Judge не сможет её использовать.
_IMPOSSIBLE_TRIGGERS = (
    # Магия без заклинания
    "взлета", "взлёта", "взлетаю", "взлетел", "парю в воздух",
    "телепорт", "оживля мёртв", "оживить мёртв",
    "останов время", "останавлива время",
    "стать богом", "становлюсь богом", "невидим без зелья", "невидим без заклин",
    # Нарушение физики
    "плыв по лав", "плыву по лав", "проход сквозь стен",
    "созда золото из воздуха",
)
_DIALOGUE_TRIGGERS = (
    '"', "«", "»", "говорю", "говорит", "спрашива", "кричу", "кричит",
    "шепчу", "шепчет", "отвеча", "молвлю",
    "обраща", "сказал", "сказала", "прошепта", "прокрича",
)
_ATTACK_TRIGGERS = (
    # Глаголы атаки в первом лице
    "атакую", "бью", "бьёт", "бить", "избиваю", "забиваю", "набрасываюсь",
    "набросилс", "набрасыв", "кидаюсь на", "бросаюсь на",
    "рублю", "режу", "режет", "сруб", "разруб",
    "стреля", "выстрел", "пускаю стрел",
    "колю", "пронзаю", "тычу", "прокалыв", "пробива",
    "убива", "прикон", "добива", "добей",
    "удар", "ударяю", "ударить",
    "пинаю", "пинаю",
    "кромс", "сруб", "сёк", "секу",
    # Продолжение боя
    "продолжаю бить", "продолжаю атак", "продолжаю руб",
    "снова бью", "ещё раз бью", "опять бью",
)
_SPELL_TRIGGERS = (
    "касту", "произнес", "произношу", "заклина", "огнен", "молни",
    "лечу заклинан", "лечу руной", "исцеля", "огненный шар", "колдую",
)
_SKILL_TRIGGERS = (
    "взлом", "прячусь", "крадусь", "перелез", "прыга", "лез",
    "проверка", "уговарива", "запугива", "проскольз", "обыщ",
    "вспомина", "вспомнить", "припоми", "анализир", "ищу следы",
    "исследу",
)
# Запрос информации — «что я вижу», «осмотрюсь», «оглядываюсь» — это НЕ движение.
# Это просьба описать то, что и так перед игроком. Отдельная категория.
_OBSERVE_TRIGGERS = (
    "что я вижу", "что вижу", "что я слышу", "что слышу",
    "осмотр", "осматрива", "оглядыва", "оглядеть",
    "какие тут", "какие здесь", "кто здесь", "кто тут",
    "что вокруг", "что в округ",
)


# Глобальный флаг — детектирован ли multi-action в фразе. Читается в step_storyteller.
_MULTI_ACTION_STATE = {"flag": False, "kinds": []}


def heuristic_classify(user_input: str) -> str:
    """Чистая Python-классификация. Заменяет LLM-вызов: моментально и надёжно."""
    lo = user_input.lower()

    has_dialog = any(t in lo for t in _DIALOGUE_TRIGGERS)
    has_attack = any(t in lo for t in _ATTACK_TRIGGERS)
    has_spell = any(t in lo for t in _SPELL_TRIGGERS)
    has_skill = any(t in lo for t in _SKILL_TRIGGERS)
    has_impossible = any(t in lo for t in _IMPOSSIBLE_TRIGGERS)
    has_observe = any(t in lo for t in _OBSERVE_TRIGGERS)

    # === MULTI-ACTION DETECTION ===
    # В D&D 5e за один ход разрешены: одна реакция + одно действие + free actions.
    # «Скажу что-то» обычно free, «атакую» — основное действие. Их можно совмещать.
    # Но мастер должен оба отыграть. Здесь мы выбираем главное действие
    # (АТАКА/ЗАКЛИНАНИЕ имеет приоритет над диалогом), а в Narrative передаём
    # флаг что нужно ещё включить реплику.
    multi_kinds = []
    if has_attack: multi_kinds.append("attack")
    if has_spell: multi_kinds.append("spell")
    if has_dialog: multi_kinds.append("dialogue")
    if has_skill: multi_kinds.append("skill")
    _MULTI_ACTION_STATE["flag"] = len(multi_kinds) > 1
    _MULTI_ACTION_STATE["kinds"] = multi_kinds
    if _MULTI_ACTION_STATE["flag"]:
        logger.info(f"[classify] Multi-action detected: {multi_kinds}. Главное — берём по приоритету.")

    # Бессмысленный ввод
    words = [w for w in lo.split() if len(w) > 1]
    if len(words) <= 1 and not (has_dialog or has_attack or has_spell or has_skill or has_impossible or has_observe):
        return "НЕПОНЯТНО"

    # Приоритет: невозможное > атака > заклинание > диалог > навык > осмотр > исследование
    if has_impossible:
        return "НЕВОЗМОЖНО"
    if has_attack:
        return "АТАКА"
    if has_spell:
        return "ЗАКЛИНАНИЕ"
    if has_dialog:
        return "ДИАЛОГ"
    if has_skill:
        return "НАВЫК"
    if has_observe:
        return "ОСМОТР"   # подкатегория ИССЛЕДОВАНИЯ — без движения
    return "ИССЛЕДОВАНИЕ"


CATEGORY_TO_ACTION = {
    "АТАКА":        "attack",
    "НАВЫК":        "skill_check",
    "ЗАКЛИНАНИЕ":   "spell",
    "ДИАЛОГ":       "dialogue",
    "ОСМОТР":       "exploration",
    "ИССЛЕДОВАНИЕ": "exploration",
    "НЕВОЗМОЖНО":   "impossible",
}


# === JSON HELPER ==========================================================

def parse_json_safe(raw: str, fallback: dict) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return fallback
    snippet = raw[start:end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", snippet))
        except json.JSONDecodeError:
            return fallback


# === PIPELINE STEPS =======================================================

# Allowed enum для intent.action — всё остальное Judge придумал
_VALID_ACTIONS = {"attack", "skill_check", "spell", "dialogue", "exploration", "impossible"}


# Русские слова в фразе → ID оружия из чарника
_WEAPON_HINTS = (
    # (фраза-триггер, id-кандидат-в-инвентаре)
    ("длинн меч", "longsword"),
    ("длинным мечом", "longsword"),
    ("меч", "longsword"),  # «меч» по умолчанию → длинный меч
    ("клинок", "longsword"),
    ("лук", "shortbow"),
    ("из лука", "shortbow"),
    ("стрел", "shortbow"),
    ("выстрел", "shortbow"),
    ("кинжал", "dagger"),
    ("ножом", "dagger"),
    ("нож ", "dagger"),
)


def normalize_weapon(intent_weapon: str | None, user_input: str) -> str | None:
    """Если Logic выдумал оружие вне whitelist — пробуем подобрать из чарника
    по словам в фразе игрока. Это спасает от 'shortsword' когда есть только longsword."""
    if intent_weapon and intent_weapon in ALLOWED_WEAPONS:
        return intent_weapon
    lo = user_input.lower()
    for hint, wid in _WEAPON_HINTS:
        if hint in lo and wid in ALLOWED_WEAPONS:
            return wid
    return intent_weapon  # отдадим как есть, валидатор затем зарубит


# Слова которые ДОЛЖНЫ быть в фразе игрока чтобы соответствующий spell применился.
# Защита от семантических галлюцинаций Logic-агента: «лечу на ракете» содержит
# слово «лечу», но это полёт, не лечение. Если игрок не сказал явно «лечусь /
# использую лечение / cure», заклинание cure_wounds выбираться не должно.
SPELL_INVOCATION_WORDS = {
    "cure_wounds":   ["лечен", "исцел", "лечусь", "лечится", "хил", "cure", "хилл"],
    "hunters_mark":  ["метк", "отмеч", "охотн", "mark"],
    "magic_missile": ["магическ стрел", "магической стрел", "магическую стрел",
                      "magic missile", "magic_missile"],
    "firebolt":      ["огнен стрел", "огненную стрел", "огнен стрелу", "firebolt"],
    "fireball":      ["огнен шар", "огненный шар", "огненным шаром", "fireball"],
    "sleep":         ["усыпля", "сон ", "sleep "],
    "light":         ["свет ", "свечен", "light "],
}


def validate_intent_against_sheet(intent: dict, category: str, user_input: str) -> dict:
    """Финальная защита от галлюцинаций Judge. Whitelist уже в его system prompt'е,
    эта функция — sanity-check на случай если модель всё равно выдумала.

    Логика:
      - action не в enum → даунгрейд по category
      - weapon вне whitelist → impossible (Judge должен был это сделать сам)
      - spell вне whitelist → impossible
      - категория НЕВОЗМОЖНО → форсим impossible
    """
    action = intent.get("action", "")

    # 1) Action должен быть из allowed enum
    if action not in _VALID_ACTIONS:
        logger.warning(f"[validate] action='{action}' не в enum, "
                       f"даунгрейд в {CATEGORY_TO_ACTION.get(category)}")
        intent["action"] = CATEGORY_TO_ACTION.get(category, "skill_check")
        action = intent["action"]

    # 2) Оружие должно быть в whitelist (Judge видел список, мог нарушить)
    if action == "attack":
        weapon_id = intent.get("weapon")
        if weapon_id and weapon_id not in ALLOWED_WEAPONS:
            logger.warning(f"[validate] weapon='{weapon_id}' нет в whitelist {list(ALLOWED_WEAPONS)} → impossible")
            intent["action"] = "impossible"
            intent["reason"] = f"У персонажа нет такого оружия. Доступно: {', '.join(ALLOWED_WEAPONS.values())}."
            intent["damage_dice"] = None

    # 3) Заклинание должно быть в whitelist
    if action == "spell":
        spell_id = intent.get("spell")
        if spell_id and spell_id not in ALLOWED_SPELLS:
            logger.warning(f"[validate] spell='{spell_id}' нет в whitelist {list(ALLOWED_SPELLS)} → impossible")
            intent["action"] = "impossible"
            spell_list = ", ".join(ALLOWED_SPELLS.values()) if ALLOWED_SPELLS else "никаких"
            intent["reason"] = f"Это заклинание не входит в репертуар персонажа. Известно: {spell_list}."
            intent["damage_dice"] = None
        elif spell_id:
            # Семантическая проверка: игрок должен явно упомянуть это заклинание.
            # «Я лечу на ракете» содержит «лечу», Judge может ошибиться spell=cure_wounds —
            # но реально игрок не вызывает заклинание лечения.
            required_words = SPELL_INVOCATION_WORDS.get(spell_id, [])
            lo = user_input.lower()
            if required_words and not any(w in lo for w in required_words):
                logger.warning(
                    f"[validate] Judge выдал spell='{spell_id}', но в фразе нет ни одного триггера "
                    f"({required_words}) → impossible (семантическая галлюцинация)"
                )
                intent["action"] = "impossible"
                intent["reason"] = (
                    "Вы не произнесли формулу заклинания и не выполнили жест — "
                    "магия не сработала."
                )
                intent["damage_dice"] = None

    # 4) Если категория — НЕВОЗМОЖНО (heuristic поймала нарушение физики), форсим
    if category == "НЕВОЗМОЖНО":
        intent["action"] = "impossible"
        intent.setdefault("reason", "Действие нарушает законы этого мира.")

    return intent


def step_parse(user_input: str, world_state: dict, category: str) -> dict:
    """Logic-агент парсит фразу в JSON-intent. Whitelist оружия/заклинаний передаётся
    прямо в system prompt — модель видит что МОЖНО, не пытается выдумать новое."""
    ws_with_char = dict(world_state)
    ws_with_char["player"] = {
        "name": CHARACTER.get("name"),
        "class": CHARACTER.get("class"),
        "level": CHARACTER.get("level"),
        "hp": f"{CHARACTER.get('hp_current')}/{CHARACTER.get('hp_max')}",
        "ac": CHARACTER.get("ac"),
    }

    # === Передаём Logic список ЖИВЫХ врагов с их HP ===
    # Без этого Logic выдумывает target=bandit_3 когда в мире уже есть bandit_2.
    # Игрок думает что бьёт того же наёмника, а на деле — новый каждый ход.
    alive_enemies = []
    for eid, e in (world_state.get("entities", {}) or {}).items():
        if isinstance(e, dict) and e.get("hp_current", 0) > 0:
            alive_enemies.append({
                "id": eid,
                "name": e.get("name", eid),
                "hp": f"{e.get('hp_current')}/{e.get('hp_max', '?')}",
            })
    if alive_enemies:
        ws_with_char["alive_enemies"] = alive_enemies

    # Динамический system prompt с whitelist'ом из чарника
    judge_sys = build_judge_system_prompt()
    # Если в бою кто-то есть — добавим явное правило об использовании их ID
    if alive_enemies:
        enemies_lines = "\n".join(
            f"  - id={e['id']}, имя='{e['name']}', HP {e['hp']}"
            for e in alive_enemies
        )
        judge_sys += (
            f"\n\nВ ТЕКУЩЕМ БОЮ есть живые враги:\n{enemies_lines}\n"
            f"Используй ИХ id в поле 'target', НЕ ВЫДУМЫВАЙ новые имена. "
            f"Если игрок говорит 'его', 'наёмник', 'противник' — это один из них."
        )

    raw = LOGIC.parse_intent(user_input, ws_with_char, sys_override=judge_sys)
    intent = parse_json_safe(raw, fallback={
        "action": CATEGORY_TO_ACTION.get(category, "skill_check"),
        "modifier": 0, "advantage": False, "disadvantage": False,
        "difficulty": "medium", "reason": "fallback",
    })

    # === FORCE action=attack если категория АТАКА а Logic ушёл не туда ===
    # Бывает: heuristic_classify уверенно дал АТАКА (есть «бью», «стреляю»),
    # но Logic вернул skill_check / spell / impossible. Если в фразе ЕСТЬ боевой
    # триггер и НЕТ магического — это точно атака, форсим.
    lo = user_input.lower()
    has_attack_trigger = any(t in lo for t in _ATTACK_TRIGGERS)
    has_spell_trigger = any(t in lo for t in _SPELL_TRIGGERS)

    needs_force = (
        category == "АТАКА"
        and has_attack_trigger
        and not has_spell_trigger
        and intent.get("action") in ("skill_check", "exploration", "spell", "impossible")
    )
    if needs_force:
        logger.warning(
            f"[validate] category=АТАКА, есть боевой триггер, нет магического, "
            f"но Logic вернул action='{intent.get('action')}' — форсим attack"
        )
        intent["action"] = "attack"
        intent["skill"] = None
        intent["spell"] = None
        # Чистим reason от старого («невозможно», «магия не сработала»)
        intent["reason"] = "Боевая атака (восстановлено по контексту фразы игрока)."
        # Подбираем оружие по фразе если Logic не подобрал
        if not intent.get("weapon") or intent.get("weapon") not in ALLOWED_WEAPONS:
            intent["weapon"] = normalize_weapon(None, user_input) or "longsword"
        # Если есть живые враги — берём первого как target
        if not intent.get("target"):
            alive_ids = [
                eid for eid, e in (world_state.get("entities", {}) or {}).items()
                if isinstance(e, dict) and e.get("hp_current", 0) > 0
            ]
            if alive_ids:
                intent["target"] = alive_ids[0]

    # === НОРМАЛИЗАЦИЯ оружия и target до валидации ===
    # Logic-агент часто выдумывает 'shortsword' когда игрок сказал «меч».
    # Сначала пробуем подобрать legit оружие из чарника по словам user_input,
    # и только если не вышло — валидатор зарубит.
    if intent.get("action") == "attack":
        intent["weapon"] = normalize_weapon(intent.get("weapon"), user_input)

    # Канонизируем target — приводим выдуманные имена к стандартным NPC-типам.
    # Если target = 'gnome_thief' / 'second_guerdian' / 'townsmith' — это всё bandit.
    if intent.get("target"):
        canonical = canonicalize_target(intent.get("target"), user_input)
        if canonical:
            intent["target"] = canonical

    # === ВАЛИДАЦИЯ против чарника (главная защита от галлюцинаций) ===
    intent = validate_intent_against_sheet(intent, category, user_input)

    # === ЗАМЕЩЕНИЕ модификатора реальным значением из чарника ===
    action = intent.get("action", "skill_check")
    if action == "attack":
        weapon = intent.get("weapon") or "longsword"
        wpn_data = get_weapon_data(weapon)
        if wpn_data:
            intent["modifier"] = wpn_data.get("attack_bonus", intent.get("modifier", 0))
            intent["damage_dice"] = wpn_data.get("damage_dice", intent.get("damage_dice"))
    elif action == "skill_check":
        skill = intent.get("skill")
        if skill:
            real_mod = get_skill_modifier(skill)
            if real_mod:
                intent["modifier"] = real_mod
    elif action == "spell":
        # Spell attack bonus = proficiency + WIS (для следопыта)
        wis_mod = CHARACTER.get("ability_modifiers", {}).get("wis", 0)
        pb = CHARACTER.get("proficiency_bonus", 2)
        intent["modifier"] = wis_mod + pb

    # Категория ОСМОТР — игрок не двигается. Сохраняем как exploration, но добавим флаг.
    if category == "ОСМОТР":
        intent["action"] = "exploration"
        intent["_observe_only"] = True

    intent.setdefault("modifier", 0)
    intent.setdefault("advantage", False)
    intent.setdefault("disadvantage", False)
    intent.setdefault("difficulty", "medium")
    return intent


def _sanitize_text(text: str) -> str:
    """Чистит выход Keeper от не-русских символов и английских слов-мусоров.
    Тот же набор замен что и в pencil_filter, но без god-modding (Keeper не пишет
    нарратив, ему это не нужно)."""
    if not isinstance(text, str):
        return text
    text = _NON_RUSSIAN_SCRIPTS.sub("", text)
    for pat, replacement in _ENGLISH_LEAK_REPLACEMENTS:
        text = pat.sub(replacement, text)
    # Особый случай: «Player» → «игрок»
    text = re.sub(r"\bPlayer\b", "игрок", text)
    text = re.sub(r"\bplayer\b", "игрок", text)
    return " ".join(text.split()).strip()


def step_state(user_input: str, narration: str, mechanics: dict, world_state: dict, intent: dict) -> dict:
    """Keeper: JSON state delta + жёсткая валидация против галлюцинаций."""
    action = intent.get("action", "")

    # === Если действие невозможно — НИЧЕГО не меняем в мире ===
    # Keeper не имеет права ставить in_flight, in_combat, transformed и т.п.
    # для действий которые не произошли.
    if action == "impossible":
        return {
            "location": world_state.get("location"),
            "summary": world_state.get("summary") or "Действие не удалось.",
            "enemies_alive": [eid for eid, e in world_state.get("entities", {}).items()
                              if e.get("hp_current", 1) > 0],
            "hp_changes": {},
            "flags": {},  # принудительно пусто
        }

    raw = LOGIC.state_delta(user_input, narration, mechanics, world_state)
    update = parse_json_safe(raw, fallback={
        "location": world_state.get("location"),
        "summary": narration[:200],
        "enemies_alive": list(world_state.get("entities", {}).keys()),
        "hp_changes": mechanics.get("state_delta", {}).get("hp", {}),
        "flags": {},
    })

    is_combat = action in ("attack", "spell") and mechanics.get("damage", 0) > 0
    if not is_combat:
        update["hp_changes"] = {}
        update["enemies_alive"] = [
            eid for eid, e in world_state.get("entities", {}).items()
            if e.get("hp_current", 1) > 0
        ]
        for flag in list(update.get("flags", {}).keys()):
            if "combat" in flag.lower() or "бой" in flag.lower():
                update["flags"].pop(flag, None)

    # Подозрительные флаги от Keeper (in_flight, transformed, и т.п. — Keeper
    # любит выдумывать). Если action не exploration/dialogue, не разрешаем флаги
    # которые подразумевают изменение состояния тела игрока.
    sus_flags = ("in_flight", "transformed", "invisible", "polymorphed", "flying")
    for flag in list(update.get("flags", {}).keys()):
        if flag.lower() in sus_flags:
            logger.warning(f"[validate-state] Keeper выдумал флаг '{flag}', удаляю")
            update["flags"].pop(flag, None)

    # ОСМОТР и dialog не меняют локацию
    if action != "exploration" or intent.get("_observe_only"):
        update["location"] = world_state.get("location", update.get("location"))

    if not is_combat and update.get("summary"):
        s = update["summary"].lower()
        if "бой" in s or "противник" in s or "врагов" in s:
            update["summary"] = narration[:150]

    # Финальная санитизация — чистим summary от не-русских символов
    # и английских слов-мусоров (Player → игрок, китайские иероглифы → удалить).
    if update.get("summary"):
        update["summary"] = _sanitize_text(update["summary"])
    if update.get("location"):
        update["location"] = _sanitize_text(update["location"])

    return update


def apply_state_delta(world_state: dict, mechanics: dict, keeper_update: dict) -> dict:
    """Применяет детерминированные изменения к world_state.
    Источник истины — mechanics (Python-движок), не Keeper.
    """
    new_state = json.loads(json.dumps(world_state))
    new_state.setdefault("entities", {})

    # Урон по НПС
    for ent_id, delta in mechanics.get("state_delta", {}).get("hp", {}).items():
        if ent_id and ent_id in new_state["entities"]:
            ent = new_state["entities"][ent_id]
            ent["hp_current"] = max(0, ent.get("hp_current", 0) + delta)
            # очищаем флаг 'spawned_turn' — НПС уже не новый в следующем ходу
            ent.pop("spawned_turn", None)

    # Урон по игроку от контратаки
    player_dmg = mechanics.get("state_delta", {}).get("player_hp")
    if player_dmg:
        new_state.setdefault("player", {})
        cur_hp = new_state["player"].get("hp_current", CHARACTER.get("hp_current", 0))
        new_state["player"]["hp_current"] = max(0, cur_hp + player_dmg)
        # Также обновляем CHARACTER чтобы /character endpoint показывал актуальное HP
        CHARACTER["hp_current"] = new_state["player"]["hp_current"]

    # Очищаем флаг spawned_turn у всех NPC после первого хода
    for ent in new_state["entities"].values():
        if isinstance(ent, dict):
            ent.pop("spawned_turn", None)

    if keeper_update.get("location"):
        new_state["location"] = keeper_update["location"]
    if keeper_update.get("summary"):
        new_state["summary"] = keeper_update["summary"]
    if keeper_update.get("flags"):
        new_state.setdefault("flags", {}).update(keeper_update["flags"])
    return new_state


# === SCHEMAS ==============================================================

class TurnRequest(BaseModel):
    user_input: str
    world_state: dict = {}
    history: list = []


class ChatRequest(BaseModel):
    mode: str
    prompt: str
    context: str = ""


# === ENDPOINTS ============================================================

@app.get("/character")
async def get_character():
    """Возвращает чарник игрока — фронт может его показать."""
    return CHARACTER


@app.post("/game/turn")
async def game_turn(req: TurnRequest):
    # Sanity-print в stdout — если этого нет в консоли, запрос не дошёл до endpoint
    print(f"[turn] >>> stdout: incoming user_input='{req.user_input[:50]}'", flush=True)
    try:
        t0 = time.time()
        user_input = req.user_input.strip()[:800]
        world_state = req.world_state or {}
        history = req.history or []
        loop = asyncio.get_event_loop()

        timings = {}

        def _tick(label, prev):
            now = time.time()
            timings[label] = round(now - prev, 2)
            logger.debug(f"    Δ {label}: {timings[label]}s  (total {round(now - t0, 2)}s)")
            return now

        logger.info(f"┌─ TURN START ─ '{user_input[:60]}'")

        # 1. Эвристическая классификация (Python, мгновенно)
        category = heuristic_classify(user_input)
        logger.info(f"[1] Classify (Python) → {category}")
        t1 = _tick("classify", t0)

        # Shortcut: бессмысленный ввод
        if category == "НЕПОНЯТНО":
            elapsed = round(time.time() - t0, 2)
            response = {
                "status": "success",
                "time": elapsed,
                "category": "НЕПОНЯТНО",
                "intent": {"action": "impossible", "reason": "Ввод не разобран"},
                "mechanics": {
                    "success": False, "roll": 0, "total": 0, "dc": 0,
                    "damage": 0, "critical": None,
                    "narration_hint": "Ввод не разобран.",
                },
                "tactic": None,
                "narration": (
                    "Мастер вопросительно смотрит на вас. — Не понял, что вы хотите сделать. "
                    "Опишите действие подробнее: куда смотрите, к кому обращаетесь, что говорите."
                ),
                "keeper_update": {
                    "location": world_state.get("location"),
                    "summary": world_state.get("summary"),
                    "enemies_alive": [], "hp_changes": {}, "flags": {},
                },
                "world_state": world_state,
            }
            logger.info(f"└─ TURN END ({elapsed}s) — НЕПОНЯТНО shortcut")
            turn_logger.info(json.dumps({
                "user_input": user_input, "category": "НЕПОНЯТНО",
                "elapsed": elapsed, "shortcut": True,
            }, ensure_ascii=False))
            return response

        # 2. Парсинг в intent (Logic LLM) — с подмесом чарника
        intent = await loop.run_in_executor(executor, step_parse, user_input, world_state, category)
        logger.info(f"[2] Parse → {json.dumps(intent, ensure_ascii=False)[:150]}")
        t2 = _tick("parse", t1)

        # 2.5. Автоспавн НПС: если игрок атакует target которого нет в мире — создаём из шаблона.
        # Это ключевой фикс — без него TacticalAI всегда видит пустой мир и НПС не контратакуют.
        if intent.get("action") in ("attack", "spell") and intent.get("target"):
            target_id = intent.get("target")
            spawned_id = ensure_npc_spawned(world_state, target_id)
            if spawned_id and spawned_id != target_id:
                logger.info(f"[2.5] Spawn → target='{target_id}' → entity '{spawned_id}' ({world_state['entities'][spawned_id].get('name')})")
                intent["target"] = spawned_id
            elif spawned_id:
                # Уже существовал — может быть с другим case или это первый спавн совпавший с canonical
                ent = world_state["entities"].get(spawned_id, {})
                if ent.get("spawned_turn"):
                    logger.info(f"[2.5] Spawn → новый НПС '{spawned_id}' ({ent.get('name')}, HP={ent.get('hp_current')})")

        # 3. Детерминированный расчёт
        mechanics = DnDEngine.resolve(intent, world_state)
        logger.info(f"[3] Engine → success={mechanics['success']} roll={mechanics['roll']} dmg={mechanics['damage']}")
        t3 = _tick("engine", t2)

        # 4. Тактик (Python) — теперь видит entities[target], составит план контратаки
        tactic = TacticalAI.plan_turn(world_state, intent, mechanics)

        # 4.5. Контратака: если NPC планирует attack — кидаем d20 за него vs player.ac,
        # считаем урон, обновляем HP игрока. Это и есть «противник бьёт в ответ».
        counter_attack = None
        if tactic and tactic.get("intent_kind") == "attack":
            attacker_id = tactic.get("monster_id")
            attacker = world_state.get("entities", {}).get(attacker_id, {})
            if attacker and attacker.get("hp_current", 0) > 0:
                player_ac = CHARACTER.get("ac", 13)
                atk_bonus = attacker.get("attack_bonus", 3)
                atk_roll, _ = DnDEngine.roll_d20()
                atk_total = atk_roll + atk_bonus
                crit_hit = (atk_roll == 20)
                crit_miss = (atk_roll == 1)
                hit = crit_hit or (not crit_miss and atk_total >= player_ac)
                dmg = 0
                if hit:
                    dmg, _ = DnDEngine.roll_dice(attacker.get("damage_dice", "1d4"))
                    if crit_hit:
                        # 5e: критический удар — удваиваем дайс-часть
                        bonus_dmg, _ = DnDEngine.roll_dice(re.sub(r"[+-]\s*\d+", "", attacker.get("damage_dice", "1d4")))
                        dmg += bonus_dmg
                counter_attack = {
                    "attacker_id": attacker_id,
                    "attacker_name": attacker.get("name", attacker_id),
                    "roll": atk_roll, "total": atk_total, "dc": player_ac,
                    "hit": hit, "critical": "hit" if crit_hit else ("miss" if crit_miss else None),
                    "damage": dmg,
                }
                # Применяем урон игроку
                if hit and dmg > 0:
                    mechanics.setdefault("state_delta", {}).setdefault("player_hp", 0)
                    mechanics["state_delta"]["player_hp"] = mechanics["state_delta"].get("player_hp", 0) - dmg
                logger.info(f"[4.5] Counter → {attacker.get('name')} d20={atk_roll}+{atk_bonus}={atk_total} vs AC{player_ac}, "
                           f"{'попал' if hit else 'промах'}, урон={dmg}")

        if tactic:
            logger.info(f"[4] Tactical → {tactic}")
        t4 = _tick("tactical" if tactic else "tactical (skip)", t3)

        # 5. Narrative (Ollama 7B) — с чарником, историей и контратакой
        narration_raw = await loop.run_in_executor(
            executor, _narrate_with_character,
            user_input, intent, mechanics, world_state, tactic, history, category,
            counter_attack,  # передаём результат контратаки чтобы Narrative его описал
        )
        narration = pencil_filter(narration_raw)
        logger.info(f"[5] Narrative → {narration[:120]}")
        t5 = _tick("narrative", t4)

        # 6. State delta (Logic LLM)
        keeper_update = await loop.run_in_executor(
            executor, step_state, user_input, narration, mechanics, world_state, intent,
        )
        logger.info(f"[6] State → {keeper_update.get('summary', '')[:80]}")
        _tick("state", t5)

        new_world = apply_state_delta(world_state, mechanics, keeper_update)

        elapsed = round(time.time() - t0, 2)
        logger.info(f"└─ TURN END ({elapsed}s)  timings={timings}")

        # Машинный лог хода
        turn_logger.info(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "user_input": user_input,
            "category": category,
            "intent": intent,
            "mechanics": mechanics,
            "tactic": tactic,
            "narration": narration,
            "keeper_update": keeper_update,
            "elapsed": elapsed,
            "timings": timings,
        }, ensure_ascii=False))

        return {
            "status": "success",
            "time": elapsed,
            "category": category,
            "intent": intent,
            "mechanics": mechanics,
            "tactic": tactic,
            "narration": narration,
            "keeper_update": keeper_update,
            "world_state": new_world,
        }
    except Exception as e:
        logger.exception(f"[ERROR /game/turn] {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _narrate_with_character(
    user_input: str, intent: dict, mechanics: dict, world_state: dict,
    tactic: Optional[dict], history: list, category: str,
    counter_attack: Optional[dict] = None,
) -> str:
    """Обёртка, которая инжектит чарник и обрабатывает category=ОСМОТР отдельно."""
    from narrative_agent import SYS_STORYTELLER

    sys_extra_parts = []

    # Усиление для ОСМОТРА — игрок НЕ двигается
    if category == "ОСМОТР" or intent.get("_observe_only"):
        sys_extra_parts.append(
            "*** РЕЖИМ ОСМОТРА ***\n"
            "Игрок СТОИТ НА МЕСТЕ и просто оглядывается. "
            "ЗАПРЕЩЕНО описывать движение игрока (он не идёт, не бежит, не выскакивает, не взбегает). "
            "Опиши ТОЛЬКО ТО, ЧТО ОН ВИДИТ ИЗ СВОЕГО ТЕКУЩЕГО ПОЛОЖЕНИЯ: помещение, предметы, "
            "других персонажей, звуки, запахи. 4-6 предложений. "
            "НЕ интерпретируй за игрока ('вы понимаете', 'вы догадываетесь', 'вы боитесь') — только то что объективно видно и слышно."
        )

    # Multi-action: в одной фразе несколько типов действий (диалог + атака и т.п.)
    if _MULTI_ACTION_STATE.get("flag"):
        kinds = _MULTI_ACTION_STATE.get("kinds", [])
        sys_extra_parts.append(
            f"*** MULTI-ACTION ***\n"
            f"Игрок в одной фразе сделал несколько действий: {', '.join(kinds)}. "
            f"Главное обработано движком (тип={intent.get('action')}). "
            f"В описании КРАТКО упомяни и другие части фразы — например, если игрок "
            f"одновременно сказал реплику и атаковал, опиши И слова И удар. "
            f"Не игнорируй второстепенные действия игрока."
        )

    # Контратака НПС: если враг ответил, обязательно описать это в нарративе.
    if counter_attack:
        attacker = counter_attack.get("attacker_name", "противник")
        if counter_attack.get("hit"):
            crit_part = " КРИТИЧЕСКИМ УДАРОМ" if counter_attack.get("critical") == "hit" else ""
            sys_extra_parts.append(
                f"*** КОНТРАТАКА ПРОТИВНИКА — ОБЯЗАТЕЛЬНО ОПИСАТЬ ***\n"
                f"После действия игрока {attacker} УСПЕШНО НАНЁС РАНУ ИГРОКУ{crit_part}. "
                f"Игрок ПОЛУЧИЛ УРОН — оружие противника достигло цели.\n"
                f"ТРЕБОВАНИЕ к нарративу:\n"
                f"1. Опиши КОНКРЕТНОЕ ПОПАДАНИЕ по игроку (клинок задел плечо, удар прошёл "
                f"по рёбрам, скользнул по руке и т.п.). НЕ пиши «свистит мимо», «промахивается» — "
                f"это НЕВЕРНО, противник попал.\n"
                f"2. Опиши физическое ощущение раны (жжение, тепло крови, боль).\n"
                f"3. БЕЗ упоминания цифр урона.\n"
                f"Эти 2-3 предложения про контратаку ОБЯЗАТЕЛЬНЫ — без них описание неполное."
            )
        else:
            crit_part = " (катастрофический промах)" if counter_attack.get("critical") == "miss" else ""
            sys_extra_parts.append(
                f"*** ОТВЕТ ПРОТИВНИКА — ПРОМАХ ***\n"
                f"После действия игрока {attacker} попытался ударить в ответ, но ПРОМАХНУЛСЯ{crit_part}. "
                f"Опиши этот неудачный выпад врага (клинок прошёл мимо, удар увернулся). "
                f"1-2 предложения."
            )

    sys_override = None
    if sys_extra_parts:
        sys_override = SYS_STORYTELLER + "\n\n" + "\n\n".join(sys_extra_parts)

    # Чарник передаётся через world_state.player_brief
    ws_extended = dict(world_state)
    ws_extended["player_brief"] = character_brief()

    return NARRATIVE.narrate(
        user_input=user_input,
        intent=intent,
        mechanics=mechanics,
        world_state=ws_extended,
        tactic=tactic,
        history=history,
        sys_override=sys_override,
    )


@app.post("/generate")
async def generate_legacy(req: ChatRequest):
    try:
        loop = asyncio.get_event_loop()
        if req.mode == "storyteller":
            text = await loop.run_in_executor(
                executor, NARRATIVE.chat_freeform,
                "Ты — Рассказчик Dark Fantasy D&D. 3-4 предложения.",
                req.prompt, 350,
            )
            return {"response": pencil_filter(text), "time": 0}
        if req.mode == "orchestrator":
            return {"response": heuristic_classify(req.prompt), "time": 0}
        if req.mode == "judge":
            return {"response": LOGIC.parse_intent(req.prompt, {}), "time": 0}
        if req.mode == "keeper":
            return {"response": LOGIC.state_delta(req.prompt, "", {}, {}), "time": 0}
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/logic")
async def process_logic(req: ChatRequest):
    return DnDEngine.process_action(req.prompt)


@app.get("/health")
async def health():
    ok, msg = NARRATIVE.verify()
    return {
        "status": "ok",
        "device": DEVICE,
        "logic_loaded": LOGIC.model is not None,
        "narrative_ready": ok,
        "narrative_msg": msg,
        "character": CHARACTER.get("name"),
        "session_log": SESSION_LOG_PATH,
        "session_turns": SESSION_TURNS_PATH,
    }


@app.get("/debug/test_log")
async def debug_test_log():
    """Пинговать руками через браузер чтобы убедиться что логгер пишет в файл.
    Если после этого вызова файл сессии пустой — значит логгер сломан."""
    print("[debug] >>> stdout: /debug/test_log called", flush=True)
    logger.info("[debug] test_log endpoint called — если эту строку нет в файле, логгер мёртв")
    turn_logger.info('{"debug": "test_log called", "ts": "' + time.strftime("%Y-%m-%dT%H:%M:%S") + '"}')
    return {
        "ok": True,
        "session_log": SESSION_LOG_PATH,
        "session_turns": SESSION_TURNS_PATH,
        "message": "Проверь оба файла — там должны появиться новые записи.",
    }


# === MAIN =================================================================

def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(start: int, end: int) -> int:
    for p in range(start, end + 1):
        if is_port_free(PYTHON_SERVER_HOST, p):
            return p
    raise RuntimeError(f"No free port in {start}..{end}")


if __name__ == "__main__":
    import uvicorn
    port = PYTHON_SERVER_PORT
    if not is_port_free(PYTHON_SERVER_HOST, port):
        port = find_free_port(port + 1, port + 20)
        print(f"[!] Port {PYTHON_SERVER_PORT} занят. Использую {port}.")
    uvicorn.run(app, host=PYTHON_SERVER_HOST, port=port)
