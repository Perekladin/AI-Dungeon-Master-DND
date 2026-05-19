# ⚡ БЫСТРЫЙ СТАРТ ОПТИМИЗИРОВАННОЙ ВЕРСИИ

## 1. Установка зависимостей (первый раз)

### Python
```bash
cd training
pip install -r requirements.txt  # если существует
# или вручную:
pip install transformers peft accelerate fastapi uvicorn torch
```

### Node.js
```bash
npm install
```

---

## 2. Запуск (2 терминала)

### Терминал 1 - ИИ сервер
```bash
cd training
python local_server.py
```
Ждите: `[+] All models loaded` (60 секунд на прогрев)

### Терминал 2 - Веб-интерфейс
```bash
npx vite --port 5173
```

---

## 3. Открыть в браузере
```
http://localhost:5173
```

---

## 4. Быстрый тест (Терминал 3)

```bash
# Тест скорости
curl -X POST http://localhost:3000/api/game/turn \
  -H "Content-Type: application/json" \
  -d '{"userInput": "Я атакую орка!"}'

# Должен ответить за 5-12 секунд
```

---

## 📊 Ожидаемые времена

| Действие | Время | Статус |
|----------|-------|--------|
| Обработка ввода | 5-12s | ✅ OK |
| Каждый агент | 1-3s | ✅ OK |
| GPU warmup | 60s | ⏳ Один раз |

---

## 🐛 Если медленно

### Проверка 1: GPU используется?
```python
# В Python:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

Если `False` - используется CPU (медленно). Нужна NVIDIA GPU.

### Проверка 2: Серверы работают?
```bash
curl http://localhost:8000/health
curl http://localhost:3000/api/health
```

### Проверка 3: Запущен benchmark
```bash
python training/benchmark.py
```

---

## 🔧 Основные оптимизации

✅ **Параллельная обработка** (4 агента одновременно)  
✅ **Короткие ответы** (40-120 токенов вместо 400)  
✅ **Greedy decoding** (вместо sampling)  
✅ **Локальные модели** (вместо Google API)  
✅ **Таймаут 12 секунд** контроль

---

## 📁 Изменённые файлы

1. `training/local_server.py` - параллельная обработка + оптимизация
2. `server.ts` - использование локального сервера вместо Gemini API
3. `src/lib/agents.ts` - новые функции для локального сервера

Смотри `OPTIMIZATION.md` для полного описания.
