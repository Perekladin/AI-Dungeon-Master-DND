"""
D&D Core AI — High-Speed Local Server.

Архитектура (соответствует PDF):
    Игрок → Orchestrator (классификация) → Judge (parser intent → JSON)
          → DnDEngine.resolve (детерминированный расчёт)
          → [Tactician (если бой)] → Storyteller (нарратив) → Keeper (JSON state delta)
          → world_state обновляется, ответ возвращается фронту.

Главные принципы:
1. LLM НЕ принимает игровых решений (это делает DnDEngine).
2. LLM возвращает ТОЛЬКО структурированный JSON либо художественный текст.
3. PENCIL-фильтр стрипает <thought> и анти-God-modding на выходе Рассказчика.
4. LRU-кэш моделей: на AMD 16GB одновременно держим максимум 3 модели.
"""

import os
import re
import sys
import json
import socket
import time
import gc
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dnd_engine import DnDEngine

app = FastAPI(title="D&D Core AI High-Speed Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

executor = ThreadPoolExecutor(max_workers=4)

PYTHON_SERVER_HOST = os.environ.get("PYTHON_SERVER_HOST", "0.0.0.0")
PYTHON_SERVER_PORT = int(os.environ.get("PYTHON_SERVER_PORT", 8000))
# Максимум моделей в VRAM одновременно. На AMD 16GB ставим 3 (4-5 не влезут без квантования).
MODEL_CACHE_MAX = int(os.environ.get("MODEL_CACHE_MAX", 3))


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


def get_torch_device(device_name: str):
    if device_name == "cuda" or device_name == "rocm":
        return torch.device("cuda")
    if device_name == "mps":
        return torch.device("mps")
    if device_name == "dml":
        import torch_directml
        return torch_directml.device()
    return torch.device("cpu")


DEVICE = get_device()
TORCH_DEVICE = get_torch_device(DEVICE)
# bitsandbytes доступен ТОЛЬКО на NVIDIA CUDA. На AMD/Intel/Apple — нет.
QUANT_AVAILABLE = (DEVICE == "cuda")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTERS_BASE_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "dnd_core_models")

MODEL_MAP = {
    "orchestrator": "Qwen/Qwen2.5-0.5B-Instruct",
    "judge":        "Qwen/Qwen2.5-1.5B-Instruct",
    "keeper":       "Qwen/Qwen2.5-1.5B-Instruct",
    "tactician":    "Qwen/Qwen2.5-3B-Instruct",
    "storyteller":  "Qwen/Qwen2.5-3B-Instruct",
}

AGENT_CONFIG = {
    # Классификация — строго детерминированно, очень короткий ответ
    "orchestrator": {"max_tokens": 16, "do_sample": False, "repetition_penalty": 1.1},
    # Судья выдаёт JSON — детерминированно, чуть больше токенов
    "judge":        {"max_tokens": 200, "do_sample": False, "repetition_penalty": 1.05},
    # Keeper тоже JSON
    "keeper":       {"max_tokens": 200, "do_sample": False, "repetition_penalty": 1.05},
    # Тактик — мини-CoT, чуть свободнее
    "tactician":    {"max_tokens": 150, "do_sample": True, "temperature": 0.6, "top_p": 0.85, "repetition_penalty": 1.1},
    # Рассказчик — литературный текст
    "storyteller":  {"max_tokens": 350, "do_sample": True, "temperature": 0.85, "top_k": 40, "top_p": 0.9, "repetition_penalty": 1.15},
}

# === MODEL LRU CACHE ======================================================

class ModelLRU:
    """OrderedDict с авто-выгрузкой старых моделей при превышении лимита VRAM."""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: "OrderedDict[str, tuple]" = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            return
        if len(self.cache) >= self.max_size:
            evict_key, evict_val = self.cache.popitem(last=False)
            print(f"[~] Выгружаю {evict_key} из VRAM, чтобы освободить место.")
            del evict_val
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.cache[key] = value

    def keys(self):
        return list(self.cache.keys())


MODEL_CACHE = ModelLRU(MODEL_CACHE_MAX)


def load_model(mode: str):
    cached = MODEL_CACHE.get(mode)
    if cached is not None:
        return cached

    base_id = MODEL_MAP.get(mode)
    if not base_id:
        raise ValueError(f"Unknown mode: {mode}")

    adapter_path = os.path.join(ADAPTERS_BASE_PATH, f"{mode}_lora_adapter")
    has_adapter = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))

    print(f"[*] Загружаю {mode} (base={base_id}, adapter={'есть' if has_adapter else 'НЕТ — используем базовую модель'})")

    quant_config = None
    if QUANT_AVAILABLE:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {
        "device_map": "auto" if DEVICE in ("cuda", "rocm") else None,
        "torch_dtype": torch.float16 if DEVICE in ("cuda", "rocm", "mps") else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    base = AutoModelForCausalLM.from_pretrained(base_id, **model_kwargs)
    if DEVICE not in ("cuda", "rocm"):
        base = base.to(TORCH_DEVICE)

    if has_adapter:
        model = PeftModel.from_pretrained(base, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    else:
        model = base
        tokenizer = AutoTokenizer.from_pretrained(base_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # warmup чтобы первый запрос не тратил CUDA initialization time
    _ = model.generate(**tokenizer("Init", return_tensors="pt").to(TORCH_DEVICE), max_new_tokens=1)

    MODEL_CACHE.put(mode, (model, tokenizer))
    return model, tokenizer


@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print(f"[i] Device: {DEVICE}  |  Torch: {torch.__version__}")
    print(f"[i] Quantization (bitsandbytes 4-bit): {'ВКЛ' if QUANT_AVAILABLE else 'ВЫКЛ (CUDA-only)'}")
    print(f"[i] LRU cache size: {MODEL_CACHE_MAX} моделей")
    if DEVICE == "cpu":
        print("[!] Warning: GPU не обнаружен. Будет ОЧЕНЬ медленно.")
    if DEVICE == "dml":
        print("[i] DirectML (AMD/Intel). Квантование недоступно — модели грузятся в FP16.")
    print("=" * 60)
    # Прогреваем 3 чаще всего используемые модели
    for mode in ["orchestrator", "judge", "storyteller"]:
        try:
            load_model(mode)
            print(f"[+] {mode} прогрет")
        except Exception as e:
            print(f"[!] Не удалось прогреть {mode}: {str(e)[:100]}")


# === SYSTEM PROMPTS =======================================================
# КРИТИЧНО: эти промпты должны совпадать с шаблоном обучения LoRA.
# Если ты переобучишь модели — обнови их и здесь, и в train_*.py.

SYS_ORCHESTRATOR = (
    "Ты классификатор намерений игрока в D&D 5e. "
    "Ответь ОДНИМ словом из набора: АТАКА, НАВЫК, ЗАКЛИНАНИЕ, ДИАЛОГ, ИССЛЕДОВАНИЕ, НЕВОЗМОЖНО. "
    "Никаких объяснений, никакого другого текста."
)

SYS_JUDGE = (
    "Ты — Судья D&D 5e. Твоя ЕДИНСТВЕННАЯ задача: распарсить русскую фразу игрока в строгий JSON.\n"
    "НЕ принимай решений об успехе/провале — это сделает движок. НЕ катай кубики — это сделает движок.\n"
    "Верни ровно один JSON-объект и ничего больше. Схема:\n"
    '{"action":"attack"|"skill_check"|"spell"|"dialogue"|"exploration"|"impossible",'
    '"target":<строка или null>,"skill":<строка или null>,"weapon":<строка или null>,'
    '"spell":<строка или null>,"modifier":<целое>,"advantage":<bool>,"disadvantage":<bool>,'
    '"difficulty":"easy"|"medium"|"hard"|null,"damage_dice":<строка типа "1d8+3" или null>,'
    '"reason":<строка>}\n'
    "Если действие физически невозможно (полёт без крыльев, телепорт без магии) — action='impossible' и заполни reason."
)

SYS_KEEPER = (
    "Ты — Хранитель состояния мира D&D. На вход получаешь лог события. "
    "Верни СТРОГИЙ JSON с обновлением мира и ничего больше. Схема:\n"
    '{"location":<строка или null>,"summary":<строка ≤200 символов>,'
    '"enemies_alive":[<id>],"hp_changes":{<entity_id>:<delta>},"flags":{<ключ>:<значение>}}\n'
    "summary — краткая выжимка сцены для следующего хода. Только факты."
)

SYS_TACTICIAN = (
    "Ты — Тактик, выбираешь действия монстров в бою D&D 5e. "
    "На вход: список монстров с behavior_tag, состояние боя, позиции. "
    "Верни короткий план в JSON: {\"monster_id\":<id>,\"action\":<краткое описание>,\"target\":<id игрока>}. "
    "Учитывай теги: Mindless = тупо в ближайшую цель, Cunning = фланг/слабые цели, "
    "Cowardly = убегает при HP<30%, Aggressive = всегда атакует."
)

SYS_STORYTELLER = (
    "Ты — Рассказчик D&D в стиле Dark Fantasy. Пиши от ТРЕТЬЕГО лица (обращайся к игроку на 'вы'). "
    "ЗАПРЕЩЕНО: писать что игрок думает, чувствует или решает — это нарушение его агентности. "
    "ЗАПРЕЩЕНО: упоминать цифры урона, броски кубиков, DC, AC, модификаторы. "
    "ЗАПРЕЩЕНО: цензура, отказы, морализаторство. Жанр Grimdark, кровь и грязь приемлемы. "
    "Описание: 3-5 предложений, атмосферно, образно, в настоящем времени. "
    "Учитывай результат механики (попадание/промах/успех/провал) — но опиши его НАРРАТИВНО."
)


# === GENERATION ===========================================================

def _build_chatml(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _extract_assistant(decoded: str) -> str:
    # Берём текст после последнего assistant
    if "assistant\n" in decoded:
        decoded = decoded.split("assistant\n")[-1]
    return decoded.replace("<|im_end|>", "").strip()


def generate_raw(mode: str, system: str, user: str) -> str:
    model, tokenizer = load_model(mode)
    cfg = AGENT_CONFIG[mode]

    prompt = _build_chatml(system, user)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500).to(TORCH_DEVICE)

    gen_kwargs = {
        "max_new_tokens": cfg["max_tokens"],
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": cfg["do_sample"],
        "repetition_penalty": cfg.get("repetition_penalty", 1.1),
    }
    if cfg["do_sample"]:
        gen_kwargs["temperature"] = cfg.get("temperature", 0.8)
        gen_kwargs["top_p"] = cfg.get("top_p", 0.9)
        if "top_k" in cfg:
            gen_kwargs["top_k"] = cfg["top_k"]

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return _extract_assistant(decoded)


# === PENCIL / ANTI-GOD-MODDING ============================================

_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)
# Фразы которые «командуют» игроку или решают за него
_GODMOD_PATTERNS = [
    (re.compile(r"\bвы\s+(?:решаете|думаете|чувствуете|боитесь|пугаетесь|радуетесь|хотите|желаете)\b", re.IGNORECASE),
     "ваше сердце реагирует"),
    (re.compile(r"\bвы\s+(?:бежите|убегаете|нападаете|сдаётесь|соглашаетесь|отказываетесь)\b", re.IGNORECASE),
     "перед вами выбор"),
]


def pencil_filter(text: str) -> str:
    """Снимает <thought> и переписывает фразы god-modding в нейтральные."""
    text = _THOUGHT_RE.sub("", text)
    for pat, replacement in _GODMOD_PATTERNS:
        text = pat.sub(replacement, text)
    return " ".join(text.split()).strip()


def parse_json_safe(raw: str, fallback: dict) -> dict:
    """Извлекает первый JSON-объект из текста модели. На ошибке возвращает fallback."""
    # Берём от первой { до последней }
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return fallback
    snippet = raw[start:end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        # Пытаемся починить трейлинг-запятые
        snippet2 = re.sub(r",\s*([}\]])", r"\1", snippet)
        try:
            return json.loads(snippet2)
        except json.JSONDecodeError:
            return fallback


# === PIPELINE STEPS =======================================================

def step_orchestrator(user_input: str) -> str:
    raw = generate_raw("orchestrator", SYS_ORCHESTRATOR, user_input)
    raw = raw.upper()
    for cat in ("АТАКА", "НАВЫК", "ЗАКЛИНАНИЕ", "ДИАЛОГ", "ИССЛЕДОВАНИЕ", "НЕВОЗМОЖНО"):
        if cat in raw:
            return cat
    return "НАВЫК"  # дефолт


def step_judge(user_input: str, world_state: dict) -> dict:
    ws_summary = json.dumps(
        {k: world_state.get(k) for k in ("location", "player", "entities", "summary") if k in world_state},
        ensure_ascii=False
    )[:1200]
    user = f"Состояние мира: {ws_summary}\nДействие игрока: {user_input}"
    raw = generate_raw("judge", SYS_JUDGE, user)
    intent = parse_json_safe(raw, fallback={
        "action": "skill_check", "modifier": 0, "advantage": False, "disadvantage": False,
        "difficulty": "medium", "reason": "fallback"
    })
    # Если LLM забыла часть полей — дозальём дефолтами
    intent.setdefault("modifier", 0)
    intent.setdefault("advantage", False)
    intent.setdefault("disadvantage", False)
    intent.setdefault("difficulty", "medium")
    return intent


def step_tactician(intent: dict, mechanics: dict, world_state: dict) -> Optional[dict]:
    """Возвращает план ответных действий монстров. Только если действие игрока было в бою."""
    if intent.get("action") not in ("attack", "spell"):
        return None
    enemies = world_state.get("entities", {})
    alive = {eid: e for eid, e in enemies.items() if e.get("hp_current", 1) > 0}
    if not alive:
        return None
    user = f"Бой. Монстры: {json.dumps(alive, ensure_ascii=False)[:600]}. " \
           f"Игрок только что: {intent.get('action')} → {'успех' if mechanics['success'] else 'провал'}."
    raw = generate_raw("tactician", SYS_TACTICIAN, user)
    return parse_json_safe(raw, fallback={"monster_id": list(alive.keys())[0], "action": "attack", "target": "player"})


def step_storyteller(user_input: str, intent: dict, mechanics: dict, world_state: dict, tactic: Optional[dict]) -> str:
    location = world_state.get("location", "неизвестное место")
    hint = mechanics.get("narration_hint", "")
    tactic_part = ""
    if tactic:
        tactic_part = f" Враг отвечает: {tactic.get('action','...')} по {tactic.get('target','игроку')}."
    user = (
        f"Локация: {location}.\n"
        f"Действие игрока: {user_input}\n"
        f"Исход механики (для контекста, НЕ цитируй цифры): {hint}.{tactic_part}\n"
        f"Опиши сцену атмосферно."
    )
    raw = generate_raw("storyteller", SYS_STORYTELLER, user)
    return pencil_filter(raw)


def step_keeper(user_input: str, narration: str, mechanics: dict, world_state: dict) -> dict:
    user = (
        f"Текущая локация: {world_state.get('location', '?')}\n"
        f"Игрок: {user_input}\n"
        f"Произошло: {narration[:400]}\n"
        f"Урон/изменения: {json.dumps(mechanics.get('state_delta', {}), ensure_ascii=False)}\n"
        "Верни JSON-обновление."
    )
    raw = generate_raw("keeper", SYS_KEEPER, user)
    return parse_json_safe(raw, fallback={
        "location": world_state.get("location"),
        "summary": narration[:200],
        "enemies_alive": list(world_state.get("entities", {}).keys()),
        "hp_changes": mechanics.get("state_delta", {}).get("hp", {}),
        "flags": {},
    })


def apply_state_delta(world_state: dict, mechanics: dict, keeper_update: dict) -> dict:
    """Применяет детерминированные изменения к world_state. Источник истины — mechanics, не Keeper."""
    new_state = json.loads(json.dumps(world_state))  # deep copy
    new_state.setdefault("entities", {})

    # HP изменения из mechanics (детерминированные)
    for ent_id, delta in mechanics.get("state_delta", {}).get("hp", {}).items():
        if ent_id and ent_id in new_state["entities"]:
            ent = new_state["entities"][ent_id]
            ent["hp_current"] = max(0, ent.get("hp_current", 0) + delta)

    # Локацию и summary разрешаем брать у Keeper (это нарративная инфа)
    if keeper_update.get("location"):
        new_state["location"] = keeper_update["location"]
    if keeper_update.get("summary"):
        new_state["summary"] = keeper_update["summary"]
    if keeper_update.get("flags"):
        new_state.setdefault("flags", {}).update(keeper_update["flags"])

    return new_state


# === API SCHEMAS ==========================================================

class ChatRequest(BaseModel):
    mode: str
    prompt: str
    context: str = ""


class TurnRequest(BaseModel):
    user_input: str
    world_state: dict = {}


# === ENDPOINTS ============================================================

@app.post("/generate")
async def generate_response(req: ChatRequest):
    """Простой endpoint — один агент, один вход. Используется фронтом для legacy-flow."""
    try:
        start = time.time()
        # Подбираем system prompt по mode
        sys_map = {
            "orchestrator": SYS_ORCHESTRATOR, "judge": SYS_JUDGE,
            "keeper": SYS_KEEPER, "tactician": SYS_TACTICIAN, "storyteller": SYS_STORYTELLER,
        }
        sys_prompt = sys_map.get(req.mode, SYS_STORYTELLER)
        user = req.prompt if not req.context else f"[Контекст]\n{req.context}\n[Запрос]\n{req.prompt}"

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(executor, generate_raw, req.mode, sys_prompt, user)

        if req.mode == "storyteller":
            raw = pencil_filter(raw)

        return {"response": raw, "time": round(time.time() - start, 2)}
    except Exception as e:
        print(f"[ERROR /generate {req.mode}] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/turn")
async def game_turn(req: TurnRequest):
    """
    Полный pipeline хода. Это РЕКОМЕНДУЕМЫЙ endpoint — фронту достаточно дёргать его.
    Внутри: orchestrator → judge (parse) → DnDEngine (compute) → tactician → storyteller → keeper.
    """
    try:
        t0 = time.time()
        user_input = req.user_input.strip()[:800]
        world_state = req.world_state or {}
        loop = asyncio.get_event_loop()

        # 1. Классификация
        category = await loop.run_in_executor(executor, step_orchestrator, user_input)
        print(f"[1] Orchestrator → {category}")

        # 2. Парсинг в intent
        intent = await loop.run_in_executor(executor, step_judge, user_input, world_state)
        print(f"[2] Judge → {json.dumps(intent, ensure_ascii=False)[:120]}")

        # 3. Детерминированный расчёт (НЕ LLM!)
        mechanics = DnDEngine.resolve(intent, world_state)
        print(f"[3] Engine → success={mechanics['success']} roll={mechanics['roll']} dmg={mechanics['damage']}")

        # 4. Тактик (только если бой)
        tactic = await loop.run_in_executor(executor, step_tactician, intent, mechanics, world_state)
        if tactic:
            print(f"[4] Tactician → {tactic}")

        # 5. Рассказчик
        narration = await loop.run_in_executor(executor, step_storyteller,
                                               user_input, intent, mechanics, world_state, tactic)
        print(f"[5] Storyteller → {narration[:80]}...")

        # 6. Keeper (JSON state delta)
        keeper_update = await loop.run_in_executor(executor, step_keeper,
                                                   user_input, narration, mechanics, world_state)
        print(f"[6] Keeper → {keeper_update.get('summary', '')[:80]}")

        # 7. Применяем детерминированные изменения
        new_world = apply_state_delta(world_state, mechanics, keeper_update)

        elapsed = round(time.time() - t0, 2)
        print(f"[✓] Turn done in {elapsed}s")

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
        print(f"[ERROR /game/turn] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/logic")
async def process_logic(req: ChatRequest):
    """Legacy-роут для старого фронта. Новый код должен звать /game/turn."""
    return DnDEngine.process_action(req.prompt)


@app.get("/health")
async def health():
    return {"status": "ok", "cached": MODEL_CACHE.keys(), "device": DEVICE, "quant": QUANT_AVAILABLE}


# === STARTUP ==============================================================

def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(start_port: int, end_port: int) -> int:
    for port in range(start_port, end_port + 1):
        if is_port_free(PYTHON_SERVER_HOST, port):
            return port
    raise RuntimeError(f"No free port in {start_port}..{end_port}")


if __name__ == "__main__":
    import uvicorn
    port = PYTHON_SERVER_PORT
    if not is_port_free(PYTHON_SERVER_HOST, port):
        port = find_free_port(port + 1, port + 20)
        print(f"[!] Port {PYTHON_SERVER_PORT} занят. Использую {port}.")
    uvicorn.run(app, host=PYTHON_SERVER_HOST, port=port)
