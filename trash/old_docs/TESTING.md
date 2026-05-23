# 🧪 Пошаговое руководство по тестированию оптимизированной версии

## ⏱️ Ожидаемые времена

| Этап | Время |
|------|-------|
| Python warmup (первый запуск) | ~60 сек |
| Средний ход игры | 5-12 сек |
| Максимум (timeout) | 12 сек |

---

## ✅ Шаг 1: Установка зависимостей

### Python
```bash
cd training
pip install -r requirements.txt
```

Если ошибка с bitsandbytes на Windows:
```bash
pip install bitsandbytes --index-url https://jllllj.github.io/bitsandbytes-windows-webui
```

### Node.js
```bash
npm install
```

---

## ✅ Шаг 2: Запуск Python сервера

**Терминал 1:**
```bash
cd training
python local_server.py
```

**Ожидаемый вывод:**
```
[!] Warming up AI engines... Please wait.
[*] Pre-loading orchestrator...
[+] Orchestrator loaded
[*] Pre-loading judge...
[+] Judge loaded
[*] Pre-loading keeper...
[+] Keeper loaded
[*] Pre-loading storyteller...
[+] Storyteller loaded
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

⏳ **Это займёт 60 секунд на первый запуск.** После этого "горячий" (модели в памяти).

---

## ✅ Шаг 3: Проверка здоровья Python сервера

**Терминал 3:**
```bash
curl http://localhost:8000/health
```

**Ожидаемый ответ:**
```json
{
  "status": "ok",
  "cached": ["orchestrator", "judge", "keeper", "storyteller"],
  "device": "cuda"
}
```

Если `device: cpu` - используется CPU (медленнее).

---

## ✅ Шаг 4: Тест одного агента

```bash
# Быстрый тест Orchestrator (самый быстрый, 0.5B модель)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "orchestrator",
    "prompt": "Я атакую орка мечом!",
    "context": ""
  }'
```

**Ожидаемый ответ (1-3 сек):**
```json
{
  "response": "АТАКА",
  "time": 0.52
}
```

---

## ✅ Шаг 5: Полный параллельный тест

```bash
# Все 4 агента одновременно
curl -X POST http://localhost:8000/game/turn \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Я атакую орка мечом! Выпало 18!",
    "world_state": {
      "player": {"name": "Warrior", "hp": 30},
      "enemies": [{"name": "Orc", "hp": 20}]
    }
  }'
```

**Ожидаемый ответ (5-8 сек):**
```json
{
  "status": "success",
  "time": 6.23,
  "orchestrator": "АТАКА",
  "judge": "УСПЕХ. Урон: 7",
  "storyteller": "Ваш меч поет...",
  "keeper": "ХП: 30, Враг: Orc 13HP"
}
```

⏱️ **Проверьте: time < 12 сек ✅**

---

## ✅ Шаг 6: Запуск Express сервера

**Терминал 2:**
```bash
npx vite --port 5173
```

**Ожидаемый вывод:**
```
  VITE v5.0.0  ready in 234 ms

  ➜  Local:   http://localhost:5173/
  ➜  press h to show help
```

---

## ✅ Шаг 7: Проверка Express API

```bash
curl http://localhost:3000/api/health
```

**Ожидаемый ответ:**
```json
{
  "status": "ok",
  "server": "running",
  "python": {
    "status": "ok",
    "cached": ["orchestrator", "judge", "keeper", "storyteller"]
  }
}
```

---

## ✅ Шаг 8: Полный игровой тест через Express API

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
  "storyteller": "Ваш клинок...",
  "keeper": "ХП: 30",
  "damage": 7,
  "timing": {
    "python_server": 6.45,
    "total_api": 6.78
  }
}
```

⏱️ **Проверьте: total_api < 12 сек ✅**

---

## ✅ Шаг 9: Открыть веб-интерфейс

```
http://localhost:5173
```

- Пишете действие игрока
- Нажимаете "Отправить" или Enter
- Ожидаете 5-12 секунд
- Видите результат в чате

---

## ✅ Шаг 10: Автоматизированный бенчмарк

```bash
python training/benchmark.py
```

**Это запустит:**
- Тест каждого агента отдельно (3 раза каждый)
- Тест всех 4 агентов параллельно
- Тест через Python /game/turn эндпоинт
- Тест через Express /api/game/turn эндпоинт

**Ожидаемый результат:**
```
============================================================
🎭 Тест отдельных агентов
============================================================

ORCHESTRATOR
Response: АТАКА
Timings: min=0.45s, avg=0.48s, max=0.52s

JUDGE
Response: УСПЕХ. Урон: 7
Timings: min=1.15s, avg=1.22s, max=1.30s

KEEPER
Response: ХП: 30
Timings: min=0.75s, avg=0.81s, max=0.88s

STORYTELLER
Response: Ваш меч поет...
Timings: min=1.90s, avg=2.05s, max=2.15s

============================================================
⚡ Тест параллельной обработки
============================================================

✓ All 4 agents processed in 2.45s
  Agent 1: 0.45s (orchestrator)
  Agent 2: 1.22s (judge)
  Agent 3: 0.81s (keeper)
  Agent 4: 2.05s (storyteller)
```

---

## 🚨 Если что-то не работает

### ❌ Python сервер не запускается

```bash
# Проверка Python версии
python --version  # Должно быть 3.8+

# Проверка зависимостей
pip list | grep torch
pip list | grep transformers
pip list | grep fastapi
```

### ❌ CUDA ошибки

```python
# Проверьте GPU
python -c "import torch; print(torch.cuda.is_available())"

# Если False - переходите на CPU
# В local_server.py DEVICE будет "cpu" автоматически
```

### ❌ Медленный ответ (>12 сек)

Проверьте что GPU используется:
```bash
nvidia-smi  # Должны видеть процесс Python
```

Если CPU - оптимизируйте дальше:
```python
# В AGENT_CONFIG уменьшите max_tokens:
"orchestrator": {"max_tokens": 20},  # было 40
"judge": {"max_tokens": 40},         # было 80
```

### ❌ Out of memory

Если VRAM < 4GB, переходите на CPU:
```python
# В local_server.py, функция get_device()
DEVICE = "cpu"  # Принудительно
```

### ❌ Express/TypeScript ошибки

```bash
# Убедитесь что Python сервер работает
curl http://localhost:8000/health

# Проверьте что Node установлен
node --version
npm --version

# Переустановите зависимости
rm -rf node_modules package-lock.json
npm install
```

---

## 📊 Интерпретация результатов

### Идеальные времена

```
Orchestrator:   0.4-0.6s ✅
Judge:          1.0-1.5s ✅
Keeper:         0.7-1.0s ✅
Storyteller:    1.8-2.5s ✅
───────────────────────
ПАРАЛЛЕЛЬНО:    ~2.5s максимум из всех
ПОЛНЫЙ ЦИКЛ:    5-12s (с Python + Express overhead)
```

### Если медленнее

| Время | Статус | Причина |
|-------|--------|---------|
| 1-5s на агент | ⚠️ Медленно | CPU вместо GPU |
| 3-5s на агент | ✅ Нормально | GPU включен |
| 15+s на ход | ❌ Слишком медленно | Timeout может срабо́тать |

---

## 🎯 Успешный тест

Всё работает если:

- ✅ Python сервер запустился с "Warming up engines"
- ✅ `curl http://localhost:8000/health` возвращает 200 OK
- ✅ Параллельный тест `/game/turn` занял < 12 сек
- ✅ `benchmark.py` показал "All 4 agents processed in ~2-3s"
- ✅ Веб-интерфейс отвечает за 5-12 сек
- ✅ `device: "cuda"` в health check

Если все ✅ - **оптимизация успешна!** 🚀

---

## 📝 Логирование для отладки

### Включить подробный лог Python

```python
# В local_server.py, добавьте:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Профилирование Python

```bash
# В отдельном терминале
python -m cProfile -s cumtime training/local_server.py
```

### Отслеживание сетевых запросов

В браузере (F12 → Network):
- `POST /api/game/turn` - должно быть < 12 сек
- `GET /api/health` - должно быть мгновенно

---

**Готово к тестированию!** 🧪✨
