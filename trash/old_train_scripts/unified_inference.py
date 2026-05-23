import torch
# Исправление ошибки torchao до импорта peft
try:
    import torchao
    from packaging import version
    if version.parse(torchao.__version__) < version.parse("0.16.0"):
        import sys
        # Временно подменяем версию, если она слишком старая для PEFT
        torchao.__version__ = "0.16.0"
except ImportError:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

import os

# Проверка Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_PATH = "/content/drive/MyDrive/dnd_core_models"
    USE_DRIVE = True
except:
    USE_DRIVE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Конфигурация соответствия адаптеров и их базовых моделей
MODEL_MAP = {
    "orchestrator": "Qwen/Qwen2.5-0.5B-Instruct",
    "judge": "Qwen/Qwen2.5-1.5B-Instruct",
    "keeper": "Qwen/Qwen2.5-1.5B-Instruct",
    "tactician": "Qwen/Qwen2.5-3B-Instruct",
    "storyteller": "Qwen/Qwen2.5-3B-Instruct"
}

def get_adapter_path(name):
    local_path = os.path.join(REPO_ROOT, "dnd_core_models", f"{name}_lora_adapter")
    drive_path = os.path.join(DRIVE_PATH, f"{name}_lora_adapter") if USE_DRIVE else None
    
    if drive_path and os.path.exists(drive_path):
        return drive_path
    return local_path

ADAPTERS = {name: get_adapter_path(name) for name in MODEL_MAP.keys()}

class DNDCoreInference:
    def __init__(self, mode=None):
        self.device = self.detect_device()
        self.torch_device = self.get_torch_device(self.device)
        self.current_mode = None
        self.model = None
        self.tokenizer = None
        if mode is not None:
            self.load_module(mode)

    def detect_device(self):
        # Prefer ROCm if present, then CUDA, MPS, DirectML, then CPU
        try:
            if getattr(torch.version, "hip", None) is not None or "rocm" in (getattr(torch.version, "__version__", "").lower()):
                return "rocm"
        except Exception:
            pass
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        try:
            import torch_directml
            return "dml"
        except Exception:
            return "cpu"

    def get_torch_device(self, device_name):
        if device_name in ["cuda", "mps"]:
            return torch.device(device_name)
        if device_name == "rocm":
            # ROCm uses the CUDA API in PyTorch; represent as 'cuda' device
            return torch.device("cuda")
        if device_name == "dml":
            import torch_directml
            return torch_directml.device()
        return torch.device("cpu")

    def _ensure_adapter_path(self, adapter_path):
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Adapter path not found: {adapter_path}. Ensure the adapter folder exists under dnd_core_models or Google Drive."
            )
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            raise FileNotFoundError(
                f"adapter_config.json not found in {adapter_path}. Check that the adapter directory is complete."
            )

    def load_module(self, mode):
        if self.current_mode == mode:
            return
        
        needed_base = MODEL_MAP.get(mode)
        if needed_base is None:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"[*] Переключение на модуль: {mode} (База: {needed_base})...")
        adapter_path = ADAPTERS.get(mode)
        self._ensure_adapter_path(adapter_path)

        if self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        model_kwargs = {
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            "device_map": "auto" if self.device in ("cuda", "rocm") else None,
        }

        base_model = AutoModelForCausalLM.from_pretrained(
            needed_base,
            **model_kwargs,
        )
        if self.device not in ("cuda", "rocm"):
            base_model = base_model.to(self.torch_device)
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.current_mode = mode
        print(f"[+] Модуль {mode} успешно загружен. (Готов отвечать на русском)")

    def generate(self, prompt, max_tokens=256):
        # Добавляем принудительную настройку на русский язык в системную часть
        # Если это инструктивная модель, Qwen понимает "Отвечай на русском"
        full_prompt = f"System: Напиши ответ на русском языке.\n{prompt}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.torch_device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1 # Немного увеличим для красоты текста
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("System:")[-1]

# Тестовый конвейер на русском
def test_pipeline():
    engine = DNDCoreInference()
    try:
        engine.load_module("orchestrator")
    except FileNotFoundError as e:
        print(f"[!] Ошибка загрузки адаптера: {e}")
        return
    
    user_input = "Я атакую орка своим огромным топором! Выпало: 18"
    
    # 1. Оркестратор (Классификация)
    intent_prompt = f"Intent Classification\nInput: {user_input}\nCategory: "
    intent = engine.generate(intent_prompt, max_tokens=10).split("Category:")[-1].strip()
    print(f"--> Определено намерение: {intent}")
    
    # 2. Судья (Анализ правил)
    if "combat" in intent.lower() or "атака" in user_input.lower():
        engine.load_module("judge")
        judge_prompt = f"System: Ты — Судья D&D 5e. Отвечай на русском.\nUser: {user_input}\nResponse: <thought>"
        rules_check = engine.generate(judge_prompt)
        print(f"--> Анализ Судьи: {rules_check}")
        
    # 3. Рассказчик (Нарратив)
    engine.load_module("storyteller")
    story_prompt = f"World Context: Подземелье, битва с орком.\nNarrative: {user_input}"
    narration = engine.generate(story_prompt)
    print(f"--> Описание мастера: {narration}")

if __name__ == "__main__":
    test_pipeline()
