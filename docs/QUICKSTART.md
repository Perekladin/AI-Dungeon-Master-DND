# D&D Core AI — Quickstart

Локальный мастер подземелий для D&D 5e. Гибридная архитектура: одна
маленькая LLM на структурированную логику + одна большая LLM на нарратив +
детерминированный Python-движок правил.

---

## Архитектура

```
Игрок
  │
  ▼
heuristic_classify  ── Python, мгновенно ─── категория действия
  │
  ▼
LogicAgent.parse_intent  ── Qwen2.5-1.5B + LoRA ─── JSON-intent
  │                                                 (с whitelist'ом из чарника)
  ▼
validate_intent  ── Python ─── защита от галлюцинаций
  │
  ▼
DnDEngine.resolve  ── Python ─── броски d20, AC/DC, урон
  │
  ▼
TacticalAI  ── Python ─── действия монстров по behavior_tag
  │
  ▼
NarrativeAgent  ── Qwen2.5-7B Q4 через Ollama ─── художественное описание
  │                                                (с чарником + историей в промпте)
  ▼
LogicAgent.state_delta  ── Qwen2.5-1.5B ─── JSON-обновление мира
  │
  ▼
apply_state_delta  ── Python ─── world_state обновлён
  │
  ▼
Игрок видит описание + персональные изменения
```

**Что отвечает за что**:

| Компонент | Реализация | Задача |
|---|---|---|
| Heuristic classify | Python-регулярки | Категоризация в АТАКА/НАВЫК/ЗАКЛИНАНИЕ/ДИАЛОГ/ОСМОТР/ИССЛЕДОВАНИЕ/НЕВОЗМОЖНО/НЕПОНЯТНО |
| Logic-агент | Qwen2.5-1.5B-Instruct + LoRA, FP16 через DirectML/transformers | Парсинг русского текста в JSON-intent, обновление JSON-состояния мира |
| DnDEngine | Чистый Python | Бросок d20, проверки AC/DC, расчёт урона, критические попадания |
| TacticalAI | Чистый Python | Поведение монстров (Mindless / Cunning / Cowardly / Aggressive / Defensive) |
| Narrative-агент | Qwen2.5-7B-Instruct Q4_K_M через Ollama (~4.5 ГБ VRAM) | Художественное описание сцены, диалоги NPC |
| Character sheet | `dnd_core_models/character_sheet.json` | Whitelist оружия/заклинаний/навыков и точные модификаторы |

---

## Требования

- **Python 3.10–3.12** (не 3.13+ — `torch-directml` ещё не имеет под него wheel-ов)
- **GPU 12+ ГБ VRAM** для разумной скорости. Тестировалось на AMD Radeon RX 9060 XT (16 ГБ)
- **Node.js 18+** для фронта
- **Ollama** — отдельная установка с https://ollama.com/download/windows

---

## Установка

### 1. Создать Python venv

```powershell
# Если у тебя несколько Python — выбери 3.12 через py launcher:
py -3.12 -m venv .venv

# Активировать venv
.\.venv\Scripts\Activate.ps1
```

### 2. Поставить Python-зависимости

```powershell
pip install -r training/requirements.txt
```

Установит:
- `torch-directml` (для AMD/Intel) или `torch` (для NVIDIA — раскомментируй в requirements)
- `transformers==4.46.3`, `peft==0.17.1`, `accelerate==1.0.1` — pinned под совместимость с torch 2.4.1
- `fastapi`, `uvicorn`, `ollama` (Python-клиент к Ollama)

### 3. Поставить Ollama и скачать модель Narrative

```powershell
# Установить Ollama: https://ollama.com/download/windows
# После установки в трее появится значок, сервис на http://localhost:11434

# Скачать модель (~4.5 ГБ)
ollama pull qwen2.5:7b-instruct-q4_K_M

# Проверить что модель установлена
ollama list
```

### 4. Поставить npm-зависимости

```powershell
npm install
```

---

## Запуск

Два процесса, два терминала.

### Терминал 1 — Python-сервер

```powershell
.\.venv\Scripts\python.exe training/local_server.py
```

При старте должно вывести:
```
[i] Device: dml  |  Torch: 2.4.1+cpu
[+] Logic-агент прогрет
[+] Narrative-агент (Ollama): Ollama OK, модель qwen2.5:7b-instruct-q4_K_M готова.
[+] Загружен чарник: Карран Тёмный (HP 24/24)
```

Сервер слушает `http://localhost:8000`.

### Терминал 2 — Frontend

```powershell
npm run dev
```

Открой `http://localhost:5173`.

---

## Кастомизация персонажа

Файл `dnd_core_models/character_sheet.json`. Здесь живёт **источник истины**:
оружие, заклинания, навыки, статы, HP, AC. Этот файл — **whitelist**: то что
здесь есть, игрок может использовать; то чего нет — невозможно.

```jsonc
{
  "name": "Карран Тёмный",
  "class": "следопыт",
  "level": 3,
  "ability_modifiers": { "str": 2, "dex": 3, "wis": 2, ... },
  "hp_current": 24, "hp_max": 24, "ac": 15,
  "weapons": [
    { "id": "longsword", "name": "длинный меч", "attack_bonus": 4, "damage_dice": "1d8+2" }
  ],
  "spells": [
    { "id": "hunters_mark", "name": "Метка охотника", "level": 1 }
  ],
  "skills": {
    "stealth": { "ability": "dex", "proficient": true, "bonus": 5 }
  }
}
```

Хочешь дать игроку **арбалет**? Добавь в `weapons`:
```json
{ "id": "crossbow", "name": "арбалет", "attack_bonus": 5, "damage_dice": "1d8+3" }
```

После рестарта сервера он автоматически появится в whitelist'е Judge,
в валидаторе и в UI чарника на фронте.

---

## Логи и отладка

Сервер пишет два файла в `training/logs/`:

| Файл | Что |
|---|---|
| `dnd_ai.log` | Текстовый лог, удобен для глазного просмотра. Rotating 5 МБ × 5 файлов |
| `turns.jsonl` | Машинно-парсимый JSON каждого хода: `user_input, intent, mechanics, narration, timings` |

Каждый ход в `dnd_ai.log` выглядит так:
```
┌─ TURN START ─ 'Я атакую гоблина'
[1] Classify (Python) → АТАКА
    Δ classify: 0.0s
[2] Parse → {"action": "attack", "weapon": "longsword", "modifier": 4, ...}
    Δ parse: 5.2s
[3] Engine → success=true roll=15 dmg=8
    Δ engine: 0.0s
[5] Narrative → Карран замахивается и наносит удар...
    Δ narrative: 18.4s
[6] State → Гоблин ранен.
    Δ state: 4.8s
└─ TURN END (28.5s)
```

---

## Скорость

| Этап | Время на AMD RX 9060 XT (DirectML, FP16) |
|---|---|
| `classify` (Python) | <0.01 сек |
| `parse` (Logic 1.5B) | 5-7 сек |
| `engine` + `tactical` (Python) | <0.01 сек |
| `narrative` (Ollama 7B Q4) | 20-40 сек |
| `state` (Logic 1.5B) | 4-6 сек |
| **Итого** | **~30-55 сек** |

На NVIDIA с CUDA-квантованием через bitsandbytes — в 3-5 раз быстрее.

---

## Переобучение Logic-агента в Google Colab (опционально)

Заводский Qwen2.5-1.5B-Instruct сам по себе справляется. Дообучение
улучшает следование схеме JSON и точность классификации.

```bash
# В Colab (T4 GPU)
!pip install -q -U bitsandbytes transformers==4.46.3 peft==0.17.1 accelerate==1.0.1 datasets trl

# Залить в /content/: colab_common.py, generate_datasets.py, train_logic.py

!python generate_datasets.py --agent logic    # ~3000 примеров
!python train_logic.py                         # обучение
```

После — скачай `logic_lora_adapter/` с Google Drive в локальный
`dnd_core_models/logic_lora_adapter/`. Сервер автоматически подхватит при
следующем запуске.

**Narrative (7B) обучать не надо** — он работает out-of-the-box через
строгий system prompt в `narrative_agent.py`.

---

## Структура проекта

```
DND_AI/
├── dnd_core_models/
│   ├── character_sheet.json    ← чарник игрока (whitelist)
│   └── logic_lora_adapter/     ← обученный LoRA-адаптер
│
├── src/                         ← React-фронт
│   ├── App.tsx
│   └── lib/agents.ts           ← клиент к /game/turn
│
├── training/                    ← Python backend
│   ├── local_server.py         ← FastAPI сервер
│   ├── logic_agent.py          ← Qwen1.5B обёртка
│   ├── narrative_agent.py      ← Ollama клиент
│   ├── tactical_ai.py          ← Python-тактик (вместо LLM)
│   ├── dnd_engine.py           ← правила D&D 5e в Python
│   ├── generate_datasets.py    ← синтетика для обучения
│   ├── train_logic.py          ← обучение Logic в Colab
│   ├── colab_common.py         ← общая инфра для тренировок
│   ├── requirements.txt
│   └── logs/                    ← логи сервера
│
├── docs/QUICKSTART.md           ← этот файл
├── package.json                 ← npm зависимости (минимум)
└── vite.config.ts               ← dev-сервер фронта
```

---

## Частые проблемы

**`No matching distribution found for torch-directml`** — у тебя Python 3.13+.
Создай venv на 3.12 через `py -3.12 -m venv .venv`.

**`Narrative-агент НЕ ГОТОВ`** при старте сервера — Ollama не запущена или
модель не скачана. Проверь иконку в трее и выполни `ollama list`.

**Сайт перезагружается во время игры** — Vite ловил изменения логов сервера.
Должно быть исправлено в `vite.config.ts` через `server.watch.ignored`. Если
повторяется — перезапусти `npm run dev`.

**Один ход занимает >60 сек** — проверь `turns.jsonl` для timing'ов. Обычно
бутылочное горлышко — `narrative` (~20-40 сек на AMD без CUDA-квантования).
Если `parse`/`state` тоже долгие — Logic выгружается из VRAM, проверь
`MODEL_CACHE_MAX` (по дефолту 1 для DirectML).

**Странные галлюцинации в описании** — посмотри `turns.jsonl`, найди ход с
проблемой. Если в `intent` есть выдуманное оружие/заклинание — это значит
пробит whitelist; либо допиши предмет в `character_sheet.json` чтобы он стал
легальным, либо добавь триггер в `_IMPOSSIBLE_TRIGGERS` для нарушений физики.
