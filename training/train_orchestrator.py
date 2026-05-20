# =================================================================
# COLAB: ВСТАВЬ В ПЕРВУЮ ЯЧЕЙКУ
# !pip install -q -U bitsandbytes transformers peft accelerate datasets trl
# !git clone https://your-repo.git || cd /content && wget <ссылки на файлы>
# Затем загрузи colab_common.py, generate_datasets.py и этот файл в /content/
# =================================================================
"""
Train: Orchestrator (Qwen2.5-0.5B).
Задача: классификация русской фразы в одну из 6 категорий.
Маленькая модель + детерминированный таргет → достаточно r=8, 1-2 эпохи.
"""

from colab_common import (
    setup_colab_env, mount_drive, save_to_drive,
    build_dataset, tokenize_function,
)
setup_colab_env()
IS_COLAB, DRIVE_PATH = mount_drive()

import os
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling

AGENT = "orchestrator"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_FILE = f"{AGENT}_data.jsonl"
OUT_DIR = f"{AGENT}_lora_adapter"

if not os.path.exists(DATA_FILE):
    print(f"[!] {DATA_FILE} не найден — генерирую из generate_datasets.py...")
    os.system(f"python generate_datasets.py --agent {AGENT}")

# === MODEL ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto",
    torch_dtype=torch.float16, low_cpu_mem_usage=True,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
for n, p in model.named_parameters():
    if p.requires_grad:
        p.data = p.data.to(torch.float32)

# === DATA ===
train_ds, eval_ds = build_dataset(DATA_FILE, AGENT, eval_ratio=0.1)
tok = tokenize_function(tokenizer, max_length=512)
train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
eval_ds = eval_ds.map(tok, batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === TRAIN ===
args = TrainingArguments(
    per_device_train_batch_size=8, gradient_accumulation_steps=2,
    num_train_epochs=2, warmup_steps=20,
    learning_rate=2e-4,
    fp16=False, bf16=False,
    logging_steps=20,
    eval_strategy="steps", eval_steps=50,
    save_strategy="steps", save_steps=100, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    output_dir=f"outputs_{AGENT}",
    optim="paged_adamw_32bit",
    report_to="none",
    gradient_checkpointing=True,
)
trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=eval_ds,
                     data_collator=collator, args=args)
trainer.train()

# === SAVE ===
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
save_to_drive(OUT_DIR, DRIVE_PATH, AGENT)
print(f"[✓] {AGENT}: обучение завершено.")
