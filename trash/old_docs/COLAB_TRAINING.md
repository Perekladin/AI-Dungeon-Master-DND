# Обучение D&D Core AI в Google Colab (Standard HF Stack)

Этот файл содержит скрипт для дообучения модели с использованием стандартных библиотек Hugging Face.

## 1. Код для Google Colab (Python)
Создайте новый Notebook в Colab, выберите **T4 GPU** и запустите:

```python
# 1. ПОДГОТОВКА СРЕДЫ (Обязательно для Colab)
# Запустите эту ячейку. Устанавливаем и обновляем библиотеки для стабильной работы.
!pip install -U bitsandbytes transformers peft accelerate datasets trl triton torchao

import os
import sys
import types
import torch
# Фикс несовместимости версий torchao
try:
    import torchao
    from packaging import version
    if version.parse(torchao.__version__) < version.parse("0.16.0"):
        torchao.__version__ = "0.16.0"
except:
    pass

from google.colab import drive

# 1.1 Монтируем Google Drive
drive.mount('/content/drive')
SAVE_PATH = "/content/drive/MyDrive/dnd_core_models"
os.makedirs(SAVE_PATH, exist_ok=True)

# Авто-настройка окружения для фикса ошибок CUDA/Triton
os.environ["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia:/usr/local/cuda/lib64"

# Заглушки для предотвращения частых ошибок импорта в Colab
try:
    import triton
    import triton.ops
except (ImportError, ModuleNotFoundError):
    if "triton" not in sys.modules: sys.modules["triton"] = types.ModuleType("triton")
    if "triton.ops" not in sys.modules: sys.modules["triton.ops"] = types.ModuleType("triton.ops")

if not torch.cuda.is_available():
    raise RuntimeError("ОШИБКА: Видеокарта не обнаружена! Перейдите в Runtime -> Change runtime type и выберите T4 GPU.")

# 2. Импорт библиотек
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# 3. Конфигурация модели (4-bit для T4 GPU)
model_id = "Qwen/Qwen2.5-3B-Instruct" 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 # Для T4 ОБЯЗАТЕЛЬНО float16
)

# 4. Загрузка
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

model = prepare_model_for_kbit_training(model)

# ПРИНУДИТЕЛЬНЫЙ ФИКС: Обучаемые веса в float32 для обхода ошибки BFloat16 на T4
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5. Форматирование данных
prompt_style = """System: You are a D&D 5e Expert.
User: {}
Response: {}"""

# ЗАГРУЗИТЕ ВАШ JSONL ФАЙЛ ПЕРЕД ЗАПУСКОМ!
try:
    dataset = load_dataset("json", data_files="your_dataset.jsonl", split="train")
except:
    print("Создаю тестовый датасет your_dataset.jsonl...")
    import json
    with open("your_dataset.jsonl", "w") as f:
        f.write(json.dumps({"input": "Hello", "output": "Greetings, hero!"}) + "\n")
    dataset = load_dataset("json", data_files="your_dataset.jsonl", split="train")

def format_prompts(batch):
    return {"text": [prompt_style.format(i, o) for i, o in zip(batch["input"], batch["output"])]}

dataset = dataset.map(format_prompts, batched=True)

# РУЧНАЯ ТОКЕНИЗАЦИЯ (Решает ошибку dataset_text_field)
def tokenize_func(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024, padding=False)

tokenized_dataset = dataset.map(tokenize_func, batched=True, remove_columns=dataset.column_names)

# 6. Запуск обучения
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=False, # Отключаем fp16 для обхода ошибки BFloat16 на T4
        bf16=False, # Отключаем bf16
        logging_steps=10,
        output_dir="outputs",
        optim="paged_adamw_32bit", # 32bit стабильнее для T4
        report_to="none",
        gradient_checkpointing=True # Экономия памяти
    ),
)

trainer.train()

# Сохраняем локально и на Google Drive
model.save_pretrained("dd_core_adapter")
tokenizer.save_pretrained("dd_core_adapter")

# Копируем на Диск для надежности
import shutil
final_drive_path = os.path.join(SAVE_PATH, "dd_core_adapter")
if os.path.exists(final_drive_path): shutil.rmtree(final_drive_path)
shutil.copytree("dd_core_adapter", final_drive_path)
print(f"--- МОДЕЛЬ СОХРАНЕНА НА GOOGLE DRIVE: {final_drive_path} ---")
```

## 2. Что делать после того, как все модели обучены?

После завершения обучения всех агентов (Judge, Keeper, Orchestrator, storyteller, tactician), у вас будет несколько папок с адаптерами. 

### Как их использовать вместе:

Вы можете загрузить их по очереди в одном скрипте. Вот пример кода для тестирования (запустите его в новой ячейке Colab):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_name = "Qwen/Qwen2.5-3B-Instruct" # Базовая модель

# 1. Загружаем базу
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Подключаем нужный адаптер (например, Судью)
# Просто укажите путь к папке, которую вы скачали/сохранили
model = PeftModel.from_pretrained(model, "judge_lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("judge_lora_adapter")

# 3. Тестируем результат
inputs = tokenizer("System: You are the Judge. Player attacks goblin with 18 roll.", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Чтобы переключиться на другую модель (н-р Storyteller):
# model = model.load_adapter("storyteller_lora_adapter", adapter_name="story")
# model.set_adapter("story")
```

### Рекомендации:
*   **Хранение**: Сохраните все папки адаптеров на Google Drive, чтобы не потерять их при закрытии сессии Colab.
*   **Имена**: Убедитесь, что в каждой папке есть файлы `adapter_config.json` и `adapter_model.safetensors` — это и есть ваш «интеллект».
*   **Версии**: Если вы обучали разные модули на разных базовых моделях (например, кого-то на 1.5B, а кого-то на 3B), вам придется загружать соответствующую базу для каждого адаптера.

## 3. Как понять, что модель обучается?

При запуске ячейки с `trainer.train()`, вы увидите таблицу. Следите за этими параметрами:
*   **Step**: Номер шага обучения. Всего их будет столько, сколько указано в `max_steps`.
*   **Training Loss**: Самый важный показатель. 
    *   Если число **уменьшается** (напр. с 1.8 до 0.5) — модель учится.
    *   Если число **скачет** (1.5 -> 0.2 -> 2.5) — `learning_rate` слишком высокий.
    *   Если число **0.000...** — модель переобучилась (будет повторять только ваши примеры).
*   **Grad Norm**: Должен быть стабильным (обычно от 0.3 до 1.5).

## 3. Специфика обучения агента «Судья»

Для «Судьи» (Logic Agent) качество данных важнее количества. Используйте следующий формат в вашем JSONL датасете:

```json
{
  "instruction": "Evaluate the D&D 5e action based on rules.",
  "input": "Player (STR 16, +3 bonus) attacks a Goblin (AC 15) with a longsword. D20 roll: 12.",
  "output": "<thought>1. Identify target AC: 15.\\n2. Calculate total roll: 12 (base) + 3 (Str bonus) + 2 (Proficiency) = 17.\\n3. Compare: 17 >= 15.\\n4. Result: Hit.</thought>{\"success\": true, \"roll\": 12, \"bonus\": 5, \"total\": 17, \"target_ac\": 15, \"damage\": 8, \"effect\": \"Longsword hit\"}"
}
```

### Золотые правила для «Судьи»:
1.  **Математика в CoT**: Всегда пишите формулу текстом в `<thought>`, прежде чем выдать цифру в JSON. Это заставляет модель «обращать внимание» на сложение.
2.  **Обработка промахов**: Добавьте 30% примеров, где действие **неуспешно** (бросок меньше AC), чтобы модель не привыкла всегда говорить "успех".
3.  **Критический провал/успех**: Добавьте примеры с натуральной 1 и 20, чтобы закрепить автоматический успех/провал.

## 4. Проверка после обучения (Inference)

Добавьте эту ячейку в конец Colab, чтобы протестировать модель:

```python
model.config.use_cache = True
model.eval()

inputs = tokenizer(
[
    prompt_style.format(
        "I try to pick the lock on a DC 15 chest. DEX +2, Prof +2. Roll: 9.",
        ""
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 256, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
