# Инструкция по локальному запуску D&D CORE V3 (Ultra Speed)

Эта версия оптимизирована для работы за **5-12 секунд**. Вся математика вынесена в скрипты, а ИИ занимается только текстом.

---

## 1. Системная подготовка (Критично для скорости!)
Если у тебя NVIDIA, AMD или CPU, выбирай пакет PyTorch в соответствии с железом.
Выполни эти команды по очереди:
```bash
# 1. Установка PyTorch
# Для NVIDIA CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Для AMD Windows (DirectML):
# Прямой установкой через PyPI пакет не найдётся, нужно использовать Microsoft wheel index.
pip install --pre torch-directml -f https://raw.githubusercontent.com/microsoft/DirectML/main/packaging/wheels/directml.html
pip install torchvision torchaudio
# Если установка не проходит из-за таймаута, попробуйте повторить команду позже или скачать wheel вручную.

# Для AMD ROCm (Linux/WSL):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.8
# Для CPU (Windows/AMD или без GPU):
pip install torch torchvision torchaudio

# 2. Установка библиотек для ИИ
pip install transformers peft accelerate fastapi uvicorn pydantic

# 3. Bitsandbytes нужен только для NVIDIA/CUDA
# Для Windows:
pip install bitsandbytes --index-url https://jllllll.github.io/bitsandbytes-windows-webui
# Для Linux/WSL:
pip install bitsandbytes
```

---

## 2. Модели
Убедись, что в папке `dnd_core_models` лежат твои адаптеры:
- `orchestrator_lora_adapter`
- `judge_lora_adapter`
- `storyteller_lora_adapter`

---

## 3. Запуск
Открой два терминала в папке проекта:

**Терминал 1 (Сервер):**
```bash
python training/local_server.py
```
*При запуске сервер "прогреет" модели. Это займет минуту, зато потом всё будет летать.*

**Терминал 2 (Интерфейс):**
```bash
npx vite --port 5173 --host
```

---

## 4. Что нового?
*   **Математика честная**: Кубики теперь кидает скрипт, а не ИИ. Никакого урона 30+ от меча.
*   **Память**: ИИ теперь помнит последние 3-4 действия и связывает их.
*   **Бандхаммер**: Попытки вызвать "самолет" или "убить босса взглядом" будут пресекаться Судьей.
*   **Только Русский**: Любой английский текст в ответах теперь жестко блокируется на уровне промптов.
