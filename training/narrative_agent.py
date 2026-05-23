"""
Narrative-агент через Ollama.

Зачем отдельный модуль: Ollama — внешний процесс с REST API, и его поведение
сильно отличается от transformers (нет device_map, нет VRAM-микроменеджмента,
нет PEFT). Изолируем эту специфику в один файл, чтобы local_server.py остался
backend-agnostic.

Используем модель: qwen2.5:7b-instruct-q4_K_M (~4.5 ГБ VRAM).
"""

import os
import re
from typing import Optional
from ollama import Client


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
NARRATIVE_MODEL = os.environ.get("NARRATIVE_MODEL", "qwen2.5:7b-instruct-q4_K_M")


SYS_STORYTELLER = (
    "ЯЗЫК ОТВЕТА — ТОЛЬКО РУССКИЙ. ИСКЛЮЧИТЕЛЬНО кириллица. "
    "НИКАКИХ китайских иероглифов (汉字), японских знаков (かな・カナ), "
    "арабской вязи, иврита, корейского хангыля. Если возникает мысль написать "
    "что-то на другом языке — переведи на русский или пропусти.\n"
    "\n"
    "Ты — Рассказчик настольной игры D&D 5e. Сеттинг — КЛАССИЧЕСКОЕ ГЕРОИЧЕСКОЕ ФЭНТЕЗИ "
    "в духе Forgotten Realms, Властелина Колец, классической D&D 5e Player's Handbook. "
    "Мир жив и красочен: солнечные таверны, базары с пряностями, дороги между деревнями, "
    "древние развалины, дружелюбные NPC и тёмные подземелья. Опасности есть, но это приключение, "
    "а не безысходность.\n"
    "\n"
    "ОБРАЩЕНИЕ К ИГРОКУ — ВАЖНЕЙШЕЕ ПРАВИЛО:\n"
    "К игроку обращайся ТОЛЬКО НА 'ВЫ' во втором лице: «вы видите», «вы слышите», "
    "«вы поднимаете меч». ИМЯ ПЕРСОНАЖА И КЛАСС используй ТОЛЬКО когда NPC говорит "
    "(в кавычках). НИКОГДА не пиши «Карран сделал...», «Следопыт чувствует...», "
    "«герой подходит...» — это запрещено. Игрок узнаёт о действиях ТОЛЬКО как «вы».\n"
    "Чарник игрока в твоём промпте — это СПРАВОЧНАЯ ИНФОРМАЦИЯ для того чтобы знать "
    "оружие и заклинания, а не повод писать о персонаже в третьем лице.\n"
    "\n"
    "СТРОГИЕ ЗАПРЕТЫ:\n"
    "1. Никаких современных предметов: телефон, провод, электричество, машина, компьютер, "
    "телевизор, радио, кино, плёнка, кабель, лампочка, батарейка, фотоаппарат, самолёт, "
    "пистолет, дробовик, базука, граната, ракета, лазер, трансформер. "
    "Используй: меч, лук, факел, свеча, кольчуга, пергамент, конь, телега, фонарь с маслом.\n"
    "2. Никаких иностранных терминов: noir, fashion, deadline, deal, OK. Только русский.\n"
    "3. НЕ ПИШИ что игрок думает, чувствует, решает, боится, хочет — это нарушает агентность игрока. "
    "Описывай только ОКРУЖЕНИЕ и ВНЕШНИЕ события.\n"
    "4. НЕ упоминай цифры: урон, броски кубиков, DC, AC, HP, модификаторы, проверки навыков.\n"
    "5. НЕ пиши служебных меток: 'Примечание автора', '---', 'Сцена X', '*курсив*'.\n"
    "6. НЕ нагромождай тёмные образы: 'кровь брызжет тёмной струёй', 'плесень', 'смрад трупов', "
    "'обугленная плоть' — это перебор для классического фэнтези. Опиши действие через детали "
    "(скрип кожи доспеха, звон стали, дым от факела), а не через шок.\n"
    "7. Каждое слово должно быть осмысленным русским. Никаких выдуманных слов "
    "('калькулятор таверны', 'трёхстопный наёмник', 'стурмул').\n"
    "\n"
    "ТОН: героический, живой, иногда юмор. NPC — живые люди с характером, а не безликие декорации. "
    "Бой — напряжённый, но не отвратительный. Смерть — да, но без смакования.\n"
    "ФОРМАТ: 3-5 предложений в настоящем времени, от ВТОРОГО лица (к игроку — на 'вы')."
)


class NarrativeAgent:
    """Тонкий клиент к Ollama. Один инстанс на весь сервер."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = NARRATIVE_MODEL):
        self.client = Client(host=host)
        self.model = model
        self._verified = False

    def verify(self) -> tuple[bool, str]:
        """Проверяет что Ollama жива и нужная модель загружена."""
        try:
            models = self.client.list()
            available = [m.model for m in models.models]
            if not any(self.model in m for m in available):
                return False, (
                    f"Модель '{self.model}' не найдена в Ollama. "
                    f"Установи её командой: ollama pull {self.model}\n"
                    f"Доступные: {available[:5]}"
                )
            self._verified = True
            return True, f"Ollama OK, модель {self.model} готова."
        except Exception as e:
            return False, (
                f"Не удалось подключиться к Ollama по адресу {OLLAMA_HOST}: {e}\n"
                f"Проверь: 1) Ollama запущен (значок в трее), 2) модель загружена `ollama list`."
            )

    def narrate(
        self,
        user_input: str,
        intent: dict,
        mechanics: dict,
        world_state: dict,
        tactic: Optional[dict] = None,
        history: Optional[list] = None,
        sys_override: Optional[str] = None,
    ) -> str:
        """Генерирует художественное описание сцены."""
        location = world_state.get("location", "неизвестное место")
        world_summary = world_state.get("summary", "")
        hint = mechanics.get("narration_hint", "")
        action = intent.get("action", "exploration")
        player_brief = world_state.get("player_brief", "")  # чарник (если передан)
        observe_only = intent.get("_observe_only", False)

        # История последних 2 ходов для continuity.
        # Фильтрация: убираем любые подозрительные не-кириллические символы
        # из history ПЕРЕД отправкой в LLM. Без этого один сорванный ход
        # с китайскими символами провоцирует петлю — модель видит мусор в
        # истории и продолжает в том же стиле.
        def _sanitize_history_line(s: str) -> str:
            # Оставляем кириллицу, латиницу, цифры, базовую пунктуацию, пробелы
            return re.sub(r"[^Ѐ-ӿԀ-ԯ a-zA-Z0-9 .,!?:;\-—«»\"'()…\n]+", "", s)

        history_block = ""
        if history:
            last = history[-2:]
            lines = []
            for h in last:
                if isinstance(h, dict):
                    pl = _sanitize_history_line((h.get("player") or "")[:120])
                    ma = _sanitize_history_line((h.get("master") or "")[:200])
                    if pl: lines.append(f"Игрок: {pl}")
                    if ma: lines.append(f"Мастер: {ma}")
            if lines:
                history_block = "ПРЕДЫДУЩИЕ ХОДЫ (ты должен продолжать эту же сцену, не начинать новую):\n" + "\n".join(lines) + "\n---\n"

        tactic_part = ""
        if tactic:
            tactic_part = f" Враг в ответ: {tactic.get('action', '...')}."

        # Тип действия → конкретная инструкция
        if action == "dialogue":
            target = intent.get("target") or "ближайший NPC"
            instr = (
                f"Игрок обращается к {target}. Опиши:\n"
                f"1) Краткую реакцию NPC (мимика, жест) — 1 предложение.\n"
                f"2) ПРЯМУЮ РЕЧЬ NPC в кавычках — 1-2 предложения, в духе средневековья.\n"
                f"3) Атмосфера момента — 1 предложение."
            )
        elif action == "impossible":
            instr = (
                "Объясни через атмосферу, почему мир НЕ ОТКЛИКНУЛСЯ на намерение игрока. "
                "Это нарушает законы реальности D&D 5e. БЕЗ упоминания цифр. 2-3 предложения."
            )
        elif action in ("attack", "spell"):
            outcome = "удар достигает цели" if mechanics.get("success") else "удар уходит мимо"
            crit = mechanics.get("critical")
            if crit == "hit":
                outcome = "удар наносит ОПУСТОШИТЕЛЬНЫЙ урон (критическое попадание)"
            elif crit == "miss":
                outcome = "сокрушительный промах, игрок открывается для контратаки"
            instr = (
                f"Опиши боевую сцену: {outcome}.{tactic_part}\n"
                f"3-5 предложений, образно, с деталями (звук, кровь, движение). БЕЗ цифр."
            )
        elif action == "skill_check":
            outcome = "получилось" if mechanics.get("success") else "не получилось"
            skill = intent.get("skill") or "проверка"
            instr = f"Опиши результат проверки навыка ({skill}): {outcome}. 3-4 предложения, через детали окружения."
        elif observe_only:
            instr = (
                "Игрок СТОИТ НА МЕСТЕ и просто осматривается. "
                "ЗАПРЕЩЕНО описывать его движение (не пишет 'вы идёте', 'вы взбегаете', 'вы выходите'). "
                "Опиши ТОЛЬКО ТО, что видно из его ТЕКУЩЕГО ПОЛОЖЕНИЯ: помещение, "
                "предметы, других персонажей, звуки, запахи. 4-5 предложений."
            )
        else:
            instr = (
                "Опиши что игрок видит, слышит, чует в этой локации. "
                "3-5 предложений, средневековое фэнтези. Никаких диалогов NPC."
            )

        # Чарник в промпт — Narrative знает кто такой игрок и что у него за оружие
        char_block = ""
        if player_brief:
            char_block = f"ИГРОК (используй имя и характеристики где уместно):\n{player_brief}\n---\n"

        user = (
            f"{char_block}"
            f"{history_block}"
            f"Локация: {location}\n"
            f"Сводка мира: {world_summary}\n"
            f"Действие игрока: {user_input}\n"
            f"Намерение (тип): {action}\n"
            f"Подсказка движка (НЕ цитируй цифры): {hint}\n"
            f"---\n"
            f"{instr}"
        )

        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_override or SYS_STORYTELLER},
                {"role": "user", "content": user},
            ],
            options={
                # Снижено с 0.75 → 0.6 чтобы модель не сваливалась в китайский
                # при генерации сложных русских конструкций.
                "temperature": 0.6,
                "top_p": 0.85,
                "repeat_penalty": 1.15,  # выше чтобы не повторять фразы из истории
                "num_predict": 350,
                "num_ctx": 4096,
                # Stop-токены: китайская пунктуация и иероглифы сразу обрывают генерацию.
                # Так Ollama не успеет насыпать мусор в ответ.
                "stop": ["，", "。", "「", "」", "『", "』", "・", "、"],
            },
        )
        return resp.message.content.strip() if resp.message.content else "..."

    def chat_freeform(self, system: str, user: str, max_tokens: int = 350) -> str:
        """Свободный чат — для отладки и кастомных запросов."""
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0.7, "num_predict": max_tokens},
        )
        return resp.message.content.strip()
