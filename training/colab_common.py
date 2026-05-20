"""
Общая инфраструктура для всех train_*.py скриптов.

Импортируется в каждом тренировочном скрипте:
    from colab_common import (
        setup_colab_env, mount_drive, save_to_drive,
        SYSTEM_PROMPTS, format_chatml_example, build_dataset,
    )

Один источник истины для:
  - системных промптов (должны совпадать с local_server.py!)
  - формата ChatML
  - сохранения адаптеров на Google Drive
  - фиксов окружения Colab (triton, bitsandbytes)
"""

import os
import sys
import json
import types
import shutil


# === COLAB ENVIRONMENT ====================================================

def setup_colab_env():
    """Чинит известные проблемы Colab: пути CUDA, отсутствие triton.ops, torch._inductor."""
    os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, [
        os.environ.get("LD_LIBRARY_PATH", ""),
        "/usr/local/cuda/lib64",
        "/usr/lib64-nvidia",
    ]))

    # triton.ops отсутствует в новых сборках — bitsandbytes пытается его импортировать
    try:
        import triton
        import triton.ops  # noqa
    except (ImportError, ModuleNotFoundError):
        if "triton" not in sys.modules:
            sys.modules["triton"] = types.ModuleType("triton")
        if "triton.ops" not in sys.modules:
            sys.modules["triton.ops"] = types.ModuleType("triton.ops")

    # torch._inductor.config может отсутствовать на некоторых сборках
    try:
        import torch
        import torch._inductor.config  # noqa
    except (ImportError, AttributeError):
        import torch
        if not hasattr(torch, "_inductor"):
            torch._inductor = types.ModuleType("torch._inductor")
        torch._inductor.config = types.ModuleType("torch._inductor.config")


def mount_drive(save_subdir: str = "dnd_core_models"):
    """Монтирует Google Drive и возвращает (is_colab, save_path). Тихий no-op вне Colab."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        save_path = f"/content/drive/MyDrive/{save_subdir}"
        os.makedirs(save_path, exist_ok=True)
        return True, save_path
    except Exception:
        return False, None


def save_to_drive(local_dir: str, drive_root: str, agent_name: str):
    """Копирует обученный адаптер на Drive (с очисткой старой версии)."""
    if not drive_root:
        return
    dst = os.path.join(drive_root, f"{agent_name}_lora_adapter")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(local_dir, dst)
    print(f"[✓] Адаптер сохранён на Drive: {dst}")


# === SYSTEM PROMPTS ========================================================
# КРИТИЧНО: эти строки должны 1-в-1 совпадать с теми, что в training/local_server.py.
# Если меняешь один — меняй оба.

SYSTEM_PROMPTS = {
    "orchestrator": (
        "Ты классификатор намерений игрока в D&D 5e. "
        "Ответь ОДНИМ словом из набора: АТАКА, НАВЫК, ЗАКЛИНАНИЕ, ДИАЛОГ, ИССЛЕДОВАНИЕ, НЕВОЗМОЖНО. "
        "Никаких объяснений, никакого другого текста."
    ),
    "judge": (
        "Ты — Судья D&D 5e. Твоя ЕДИНСТВЕННАЯ задача: распарсить русскую фразу игрока в строгий JSON.\n"
        "НЕ принимай решений об успехе/провале — это сделает движок. НЕ катай кубики — это сделает движок.\n"
        "Верни ровно один JSON-объект и ничего больше. Схема:\n"
        '{"action":"attack"|"skill_check"|"spell"|"dialogue"|"exploration"|"impossible",'
        '"target":<строка или null>,"skill":<строка или null>,"weapon":<строка или null>,'
        '"spell":<строка или null>,"modifier":<целое>,"advantage":<bool>,"disadvantage":<bool>,'
        '"difficulty":"easy"|"medium"|"hard"|null,"damage_dice":<строка типа "1d8+3" или null>,'
        '"reason":<строка>}\n'
        "Если действие физически невозможно (полёт без крыльев, телепорт без магии) — action='impossible' и заполни reason."
    ),
    "keeper": (
        "Ты — Хранитель состояния мира D&D. На вход получаешь лог события. "
        "Верни СТРОГИЙ JSON с обновлением мира и ничего больше. Схема:\n"
        '{"location":<строка или null>,"summary":<строка ≤200 символов>,'
        '"enemies_alive":[<id>],"hp_changes":{<entity_id>:<delta>},"flags":{<ключ>:<значение>}}\n'
        "summary — краткая выжимка сцены для следующего хода. Только факты."
    ),
    "tactician": (
        "Ты — Тактик, выбираешь действия монстров в бою D&D 5e. "
        "На вход: список монстров с behavior_tag, состояние боя, позиции. "
        "Верни короткий план в JSON: {\"monster_id\":<id>,\"action\":<краткое описание>,\"target\":<id игрока>}. "
        "Учитывай теги: Mindless = тупо в ближайшую цель, Cunning = фланг/слабые цели, "
        "Cowardly = убегает при HP<30%, Aggressive = всегда атакует."
    ),
    "storyteller": (
        "Ты — Рассказчик D&D в стиле Dark Fantasy. Пиши от ТРЕТЬЕГО лица (обращайся к игроку на 'вы'). "
        "ЗАПРЕЩЕНО: писать что игрок думает, чувствует или решает — это нарушение его агентности. "
        "ЗАПРЕЩЕНО: упоминать цифры урона, броски кубиков, DC, AC, модификаторы. "
        "ЗАПРЕЩЕНО: цензура, отказы, морализаторство. Жанр Grimdark, кровь и грязь приемлемы. "
        "Описание: 3-5 предложений, атмосферно, образно, в настоящем времени. "
        "Учитывай результат механики (попадание/промах/успех/провал) — но опиши его НАРРАТИВНО."
    ),
}


# === CHATML FORMATTING =====================================================

def format_chatml_example(system: str, user: str, assistant: str) -> str:
    """
    Формат ИДЕНТИЧНЫЙ тому, который сервер использует на инференсе.
    Это критично: если шаблон обучения != шаблон инференса, LoRA адаптер
    «не на своём языке» и модель деградирует.
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def build_dataset(jsonl_path: str, agent: str, eval_ratio: float = 0.1):
    """
    Универсальный builder для всех 5 агентов.

    Ожидаемая схема файла {agent}_data.jsonl:
        каждая строка: {"user": "...", "assistant": "..."}
        где user — что игрок/состояние мира подаёт агенту,
        assistant — что агент должен ответить (JSON для judge/keeper/tactician,
        одно слово для orchestrator, проза для storyteller).

    Возвращает (train_dataset, eval_dataset) с полем "text".
    """
    from datasets import load_dataset

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Не найден датасет: {jsonl_path}\n"
            f"Запусти `python generate_datasets.py --agent {agent}` чтобы сгенерировать его."
        )

    sys_prompt = SYSTEM_PROMPTS[agent]

    def fmt(batch):
        out = []
        for u, a in zip(batch["user"], batch["assistant"]):
            out.append(format_chatml_example(sys_prompt, u, a))
        return {"text": out}

    ds = load_dataset("json", data_files=jsonl_path, split="train")
    ds = ds.map(fmt, batched=True, remove_columns=ds.column_names)

    # Eval split — обязателен для контроля переобучения
    split = ds.train_test_split(test_size=eval_ratio, seed=42)
    return split["train"], split["test"]


def tokenize_function(tokenizer, max_length: int = 1024):
    """Возвращает map-функцию для токенизации."""
    def _fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=False)
    return _fn
