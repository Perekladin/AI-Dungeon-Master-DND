"""
Logic-агент: объединяет Orchestrator + Judge + Keeper в одной модели Qwen2.5-1.5B.

Зачем: эти три задачи — разновидности structured output. Старая архитектура
тратила 3 LoRA-адаптера и 3 inference-вызова на то, что одна модель легко
делает с разными system-prompt'ами. Это экономит VRAM, упрощает обучение
(один датасет смешанный) и снимает рассинхрон между агентами.

Роли:
    role="classify"   — orchestrator: ОДНО СЛОВО из 6 категорий
    role="parse"      — judge: JSON-intent для DnDEngine
    role="state"      — keeper: JSON state delta

ChatML-формат идентичен серверу. LoRA-адаптер обучается на смеси трёх задач —
модель сама различает их по system-prompt'у.
"""

import os
import json
import gc
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# === SYSTEM PROMPTS — точно те же, что в обучающем датасете ===

SYS_CLASSIFY = (
    "Ты классификатор намерений игрока в D&D 5e. "
    "Ответь ОДНИМ словом из набора: АТАКА, НАВЫК, ЗАКЛИНАНИЕ, ДИАЛОГ, ИССЛЕДОВАНИЕ, НЕВОЗМОЖНО. "
    "Никаких объяснений, никакого другого текста."
)

SYS_PARSE = (
    "Ты — Судья D&D 5e. Твоя ЕДИНСТВЕННАЯ задача: распарсить русскую фразу игрока в строгий JSON.\n"
    "НЕ принимай решений об успехе/провале — это сделает движок. НЕ катай кубики — это сделает движок.\n"
    "Верни ровно один JSON-объект и ничего больше. Схема:\n"
    '{"action":"attack"|"skill_check"|"spell"|"dialogue"|"exploration"|"impossible",'
    '"target":<строка или null>,"skill":<строка или null>,"weapon":<строка или null>,'
    '"spell":<строка или null>,"modifier":<целое>,"advantage":<bool>,"disadvantage":<bool>,'
    '"difficulty":"easy"|"medium"|"hard"|null,"damage_dice":<строка типа "1d8+3" или null>,'
    '"reason":<строка>}\n'
    "Если действие физически невозможно — action='impossible' и заполни reason."
)

SYS_STATE = (
    "Ты — Хранитель состояния мира D&D. На вход получаешь лог события. "
    "Верни СТРОГИЙ JSON с обновлением мира и ничего больше. Схема:\n"
    '{"location":<строка или null>,"summary":<строка ≤200 символов>,'
    '"enemies_alive":[<id>],"hp_changes":{<entity_id>:<delta>},"flags":{<ключ>:<значение>}}\n'
    "summary — краткая выжимка сцены для следующего хода. Только факты."
)

ROLE_TO_SYS = {
    "classify": SYS_CLASSIFY,
    "parse":    SYS_PARSE,
    "state":    SYS_STATE,
}

ROLE_TO_MAX_TOKENS = {
    "classify": 16,
    "parse":    220,
    "state":    220,
}


class LogicAgent:
    """Singleton-стиль: одна модель загружена один раз, переиспользуется для всех ролей."""

    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        torch_device=None,
        quant_available: bool = True,
    ):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.device = device
        self.torch_device = torch_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quant_available = quant_available
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            return
        print(f"[*] LogicAgent загружает {self.base_model_id}"
              f"{' + LoRA' if self.adapter_path else ' (без адаптера)'}")

        quant_config = None
        if self.quant_available:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model_kwargs = {
            "device_map": "auto" if self.device in ("cuda", "rocm") else None,
            "torch_dtype": torch.float16 if self.device in ("cuda", "rocm", "mps", "dml") else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        base = AutoModelForCausalLM.from_pretrained(self.base_model_id, **model_kwargs)
        if self.device not in ("cuda", "rocm"):
            base = base.to(self.torch_device)

        has_adapter = self.adapter_path and os.path.exists(
            os.path.join(self.adapter_path, "adapter_config.json"))
        if has_adapter:
            self.model = PeftModel.from_pretrained(base, self.adapter_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        else:
            self.model = base
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Warmup
        _ = self.model.generate(
            **self.tokenizer("Init", return_tensors="pt").to(self.torch_device),
            max_new_tokens=1,
        )
        print(f"[+] LogicAgent готов")

    def unload(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(self, role: str, user: str, sys_override: Optional[str] = None) -> str:
        """sys_override — динамический system prompt (например для whitelist из чарника).
        Если не задан, берётся статичный ROLE_TO_SYS[role]."""
        self.load()
        sys_prompt = sys_override or ROLE_TO_SYS[role]
        max_tokens = ROLE_TO_MAX_TOKENS[role]

        prompt = (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500).to(self.torch_device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.05 if role != "classify" else 1.1,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "assistant\n" in decoded:
            decoded = decoded.split("assistant\n")[-1]
        return decoded.replace("<|im_end|>", "").strip()

    # === Public API: три роли ===

    def classify(self, user_input: str) -> str:
        return self._generate("classify", user_input)

    def parse_intent(self, user_input: str, world_state: dict, sys_override: Optional[str] = None) -> str:
        ws_summary = json.dumps(
            {k: world_state.get(k) for k in ("location", "player", "entities", "summary") if k in world_state},
            ensure_ascii=False,
        )[:1200]
        user = f"Состояние мира: {ws_summary}\nДействие игрока: {user_input}"
        return self._generate("parse", user, sys_override=sys_override)

    def state_delta(self, user_input: str, narration: str, mechanics: dict, world_state: dict) -> str:
        user = (
            f"Текущая локация: {world_state.get('location', '?')}\n"
            f"Игрок: {user_input}\n"
            f"Произошло: {narration[:400]}\n"
            f"Урон/изменения: {json.dumps(mechanics.get('state_delta', {}), ensure_ascii=False)}\n"
            "Верни JSON-обновление."
        )
        return self._generate("state", user)
