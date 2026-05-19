# --- ДЛЯ ЗАПУСКА В COLAB (ПЕРВАЯ ЯЧЕЙКА) ---
# !pip install -q -U bitsandbytes transformers peft accelerate datasets trl triton
# import os
# os.environ["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia"
# -------------------------------------------

import torch
import types
import sys
import os

# --- ИНТЕГРАЦИЯ С GOOGLE DRIVE ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_PATH = "/content/drive/MyDrive/dnd_core_models"
    os.makedirs(SAVE_PATH, exist_ok=True)
    IS_COLAB = True
except:
    IS_COLAB = False
# ---------------------------------

# --- ИСПРАВЛЕНИЕ ОШИБОК ИМПОРТА (Triton / bitsandbytes) ---
def patch_environment():
    os.environ["LD_LIBRARY_PATH"] = ":".join([os.environ.get("LD_LIBRARY_PATH", ""), "/usr/local/cuda/lib64", "/usr/lib64-nvidia"])
    try:
        import triton
        import triton.ops
    except (ImportError, ModuleNotFoundError):
        if "triton" not in sys.modules: sys.modules["triton"] = types.ModuleType("triton")
        if "triton.ops" not in sys.modules: sys.modules["triton.ops"] = types.ModuleType("triton.ops")
    try:
        import torch._inductor.config
    except (ImportError, AttributeError):
        if not hasattr(torch, "_inductor"): torch._inductor = types.ModuleType("torch._inductor")
        torch._inductor.config = types.ModuleType("torch._inductor.config")

patch_environment()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# Конфигурация: Keeper
model_id = "Qwen/Qwen2.5-1.5B-Instruct" 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 # Обязательно для T4
)

# Загрузка
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Подготовка LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=12,
    lora_alpha=24,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ФИКС ДЛЯ T4
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

# Хранитель (Keeper) - JSON update
prompt_style = "Log: {}\nSystem: Update JSON State.\nNew State: {}"

def format_prompts(batch):
    return { "text": [prompt_style.format(l, s) for l, s in zip(batch["log"], batch["state"])] }

# Загрузка данных
import os
import json

file_path = "keeper_data.jsonl"

if not os.path.exists(file_path):
    print(f"\n[!] ФАЙЛ {file_path} НЕ НАЙДЕН. Создаю пример данных...")
    dummy_data = [{"log": "Goblin dies", "state": "{}"}]
    with open(file_path, "w") as f:
        for entry in dummy_data: f.write(json.dumps(entry) + "\n")

try:
    dataset = load_dataset("json", data_files=file_path, split="train")
    dataset = dataset.map(format_prompts, batched=True)
except Exception as e:
    raise RuntimeError(f"Ошибка: {e}")

# РУЧНАЯ ТОКЕНИЗАЦИЯ
def tokenize_func(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024, padding=False)

tokenized_dataset = dataset.map(tokenize_func, batched=True, remove_columns=dataset.column_names)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=1e-4,
        fp16=False,
        bf16=False,
        logging_steps=10,
        output_dir="outputs_keeper",
        optim="paged_adamw_32bit",
        report_to="none",
        gradient_checkpointing=True
    ),
)

trainer.train()
model.save_pretrained("keeper_lora_adapter")
tokenizer.save_pretrained("keeper_lora_adapter")

# --- КОПИРОВАНИЕ НА DRIVE ---
if IS_COLAB:
    import shutil
    drive_adapter_path = os.path.join(SAVE_PATH, "keeper_lora_adapter")
    if os.path.exists(drive_adapter_path): shutil.rmtree(drive_adapter_path)
    shutil.copytree("keeper_lora_adapter", drive_adapter_path)
    print(f"--- Адаптер сохранен на Google Drive: {drive_adapter_path} ---")
