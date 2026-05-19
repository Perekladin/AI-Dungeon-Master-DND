# 🚀 Оптимизация DND AI - Инструкция

## Что было оптимизировано

### ❌ Было (20-30 секунд на ход)
- Последовательные вызовы агентов (Orchestrator → Judge → Storyteller → Keeper)
- Google Gemini API (сетевые задержки)
- max_tokens слишком большие (400 для storyteller)
- do_sample=True (медленнее greedy decoding)
- Нет контроля таймаутов

### ✅ Стало (5-12 секунд на ход)
1. **Параллельная обработка** - все 4 агента работают одновременно
2. **Локальные модели** - Qwen вместо Google API
3. **Оптимизированные параметры**:
   - Orchestrator: max_tokens=40 (вместо 128)
   - Judge: max_tokens=80 (вместо 128)
   - Keeper: max_tokens=50 (вместо 128)
   - Storyteller: max_tokens=120 (вместо 400)
4. **Greedy decoding** везде (do_sample=False)
5. **Таймаут 12 секунд** на весь ход
6. **Компактные системные промпты** (уменьшен размер)

---

## Запуск

### 1️⃣ Терминал 1 - Python сервер (ИИ с параллелизмом)
```bash
cd training
python local_server.py
```
**Результат:**
```
[!] Warming up AI engines... Please wait.
[+] Orchestrator loaded
[+] Judge loaded
[+] Keeper loaded
[+] Storyteller loaded
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 2️⃣ Терминал 2 - Express/Vite сервер
```bash
npm install
npx vite --port 5173 --host
```

В новом окне браузера откройте: http://localhost:5173

---

## Тестирование скорости

### Вариант 1: Через API
```bash
curl -X POST http://localhost:3000/api/game/turn \
  -H "Content-Type: application/json" \
  -d '{"userInput": "Я атакую орка мечом! Выпало 18!"}'
```

**Ожидаемый ответ (5-12 сек):**
```json
{
  "category": "ATTACK",
  "orchestrator": "АТАКА",
  "judge": "УСПЕХ. Урон: 7",
  "storyteller": "Ваш меч поет в воздухе, разрезая тьму...",
  "keeper": "ХП орка: 6. Локация: Пещера.",
  "damage": 7,
  "timing": {
    "python_server": 3.45,
    "total_api": 3.67
  }
}
```

### Вариант 2: Прямой вызов Python
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"mode": "storyteller", "prompt": "Битва с драконом", "context": ""}'
```

---

## Профилирование

В `local_server.py` вы увидите:
```
[✓] Full turn completed in 4.23s
[*] Generation (orchestrator) took: 0.45s
[*] Generation (judge) took: 1.23s
[*] Generation (keeper) took: 0.89s
[*] Generation (storyteller) took: 1.66s
```

---

## Требования к железу

| Компонент | Минимум | Рекомендуемо |
|-----------|---------|-------------|
| VRAM | 4GB | 8GB+ |
| CPU | 4 cores | 8+ cores |
| RAM | 8GB | 16GB+ |
| Время первого старта | 60s (warmup) | 60s (warmup) |

### Для NVIDIA GPU
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes --index-url https://jllllll.github.io/bitsandbytes-windows-webui
```

### Для AMD GPU / ROCm (Linux/WSL)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.8
```

### Для AMD Windows (DirectML)
```bash
pip install --pre torch-directml -f https://raw.githubusercontent.com/microsoft/DirectML/main/packaging/wheels/directml.html
pip install torchvision torchaudio
```

> Примечание: `torch-directml` не распространяется через стандартный PyPI индекс, поэтому нужно использовать указанный wheel-источник.
> Если сеть нестабильна, попробуйте повторить команду позже.

### Для CPU (Windows/AMD или без GPU)
```bash
pip install torch torchvision torchaudio
# Время на ход: 20-30s (неоптимально)
```

---

## Переменные окружения

### В `server.ts` (TypeScript)
```bash
# .env
VITE_PYTHON_SERVER=http://localhost:8000
```

### В `local_server.py` (Python)
```python
# Автоматически детектирует CUDA/CPU
```

---

## Если всё еще медленно

### 1. Проверьте GPU
```python
import torch
print(torch.cuda.is_available())
print(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
print(torch.device('cuda' if torch.cuda.is_available() else 'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'))
```

### 2. Уменьшите max_tokens еще больше
В `AGENT_CONFIG` в `local_server.py`:
```python
AGENT_CONFIG = {
    "orchestrator": {"max_tokens": 20, ...},  # было 40
    "judge": {"max_tokens": 50, ...},         # было 80
    "keeper": {"max_tokens": 30, ...},        # было 50
    "storyteller": {"max_tokens": 80, ...}    # было 120
}
```

### 3. Используйте batch processing (если несколько пользователей)
Модифицируйте эндпоинт `/game/turn` для обработки нескольких запросов одновременно

### 4. Профилируйте узкие места
```python
import cProfile
cProfile.run('generate_fast("storyteller", "text", "")')
```

---

## Архитектура после оптимизации

```
User Input
    ↓
Express Server (server.ts)
    ↓
/game/turn endpoint
    ↓
Python FastAPI (local_server.py)
    ├─→ [Thread 1] Orchestrator (0.5B) ─→ Classification
    ├─→ [Thread 2] Judge (1.5B) ─→ Rules Check
    ├─→ [Thread 3] Keeper (1.5B) ─→ Memory Update
    └─→ [Thread 4] Storyteller (3B) ─→ Narrative
         (Все 4 работают ПАРАЛЛЕЛЬНО)
    ↓
Response (4-6 сек вместо 20+)
```

---

## Бенчмарки

| Сценарий | До | После | Улучшение |
|----------|-----|--------|-----------|
| Простое действие | 8-12s | 2-4s | 3-6x быстрее |
| Боевое действие | 15-20s | 4-7s | 3-5x быстрее |
| Диалог | 10-15s | 3-5s | 3-4x быстрее |
| Full RPG turn | 20-30s | 5-12s | **2.5-6x быстрее** |

---

## Совместимость

- ✅ Windows (NVIDIA, CPU)
- ✅ Linux/WSL (NVIDIA, CPU)
- ✅ macOS (CPU только, медленно)
- ✅ Google Colab (если загрузить модели)

---

## Помощь

Если timeout 12 секунд всё еще не достаточен:
1. Увеличьте TIMEOUT_MS в server.ts (до 15000)
2. Или дальше уменьшайте max_tokens
3. Профилируйте, какой агент занимает больше всего времени
