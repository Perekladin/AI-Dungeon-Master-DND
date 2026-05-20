"""
Детерминированный движок правил D&D 5e.

Контракт: LLM-Судья отвечает ТОЛЬКО за парсинг русского текста игрока в структурированный intent (JSON).
Все игровые расчёты (d20, AC, DC, урон, эффекты) выполняются ЗДЕСЬ, в Python.
Это гарантирует Rule Fidelity = 100% и устраняет галлюцинации модели.

Главный API:
    DnDEngine.resolve(intent: dict, world_state: dict) -> dict
"""

import random
import re
from typing import Optional


# Карта проверок навыков -> характеристика
SKILL_TO_ABILITY = {
    "athletics": "str", "атлетика": "str",
    "acrobatics": "dex", "акробатика": "dex",
    "sleight_of_hand": "dex", "ловкость_рук": "dex",
    "stealth": "dex", "скрытность": "dex",
    "arcana": "int", "магия": "int",
    "history": "int", "история": "int",
    "investigation": "int", "анализ": "int",
    "nature": "int", "природа": "int",
    "religion": "int", "религия": "int",
    "animal_handling": "wis", "уход_за_животными": "wis",
    "insight": "wis", "проницательность": "wis",
    "medicine": "wis", "медицина": "wis",
    "perception": "wis", "внимательность": "wis",
    "survival": "wis", "выживание": "wis",
    "deception": "cha", "обман": "cha",
    "intimidation": "cha", "запугивание": "cha",
    "performance": "cha", "выступление": "cha",
    "persuasion": "cha", "убеждение": "cha",
}

# Стандартные DC для типичных действий (если LLM не указала)
DEFAULT_DC = {
    "trivial": 5,
    "easy": 10,
    "medium": 15,
    "hard": 20,
    "very_hard": 25,
    "nearly_impossible": 30,
}


def ability_modifier(score: int) -> int:
    """D&D 5e: модификатор = (score - 10) // 2, округление вниз."""
    return (score - 10) // 2


class DnDEngine:
    @staticmethod
    def roll_d20(advantage: bool = False, disadvantage: bool = False) -> tuple[int, list[int]]:
        """Возвращает (итог, [все_броски]). Преимущество = берём максимум из 2-х."""
        if advantage and disadvantage:
            advantage = disadvantage = False  # взаимно компенсируются по 5e
        rolls = [random.randint(1, 20)]
        if advantage or disadvantage:
            rolls.append(random.randint(1, 20))
        chosen = max(rolls) if advantage else (min(rolls) if disadvantage else rolls[0])
        return chosen, rolls

    @staticmethod
    def roll_dice(dice_str: str) -> tuple[int, list[int]]:
        """Парсит '1d8+3', '2d6', '1d4-1'. Возвращает (сумма, отдельные_броски)."""
        match = re.match(r"\s*(\d+)\s*d\s*(\d+)\s*([+-]\s*\d+)?", dice_str.strip(), re.IGNORECASE)
        if not match:
            return 0, []
        num, sides, mod = match.groups()
        rolls = [random.randint(1, int(sides)) for _ in range(int(num))]
        total = sum(rolls)
        if mod:
            total += int(mod.replace(" ", ""))
        return total, rolls

    @staticmethod
    def resolve(intent: dict, world_state: Optional[dict] = None) -> dict:
        """
        Главная функция. Принимает структурированный intent от Судьи-LLM и считает результат.

        intent format:
            {
              "action": "attack" | "skill_check" | "spell" | "dialogue" | "exploration" | "impossible",
              "actor": "player" | "monster_id",
              "target": str | None,
              "skill": str | None,          # для skill_check, e.g. "stealth", "athletics"
              "weapon": str | None,         # для attack, e.g. "longsword"
              "spell": str | None,          # для spell
              "modifier": int,              # суммарный бонус игрока (proficiency + ability)
              "advantage": bool,
              "disadvantage": bool,
              "difficulty": "easy"|"medium"|"hard"|None,   # подсказка от Судьи
              "damage_dice": str | None,    # e.g. "1d8+3"
            }

        Возвращает:
            {
              "success": bool, "roll": int, "all_rolls": [int],
              "total": int, "dc": int, "damage": int, "damage_rolls": [int],
              "critical": "hit"|"miss"|None,
              "narration_hint": str,        # подсказка Рассказчику что произошло
              "state_delta": dict,          # как изменить world_state
            }
        """
        world_state = world_state or {}
        action = intent.get("action", "skill_check")
        modifier = int(intent.get("modifier", 0))
        advantage = bool(intent.get("advantage", False))
        disadvantage = bool(intent.get("disadvantage", False))

        # Действие невозможно — нет броска
        if action == "impossible":
            return {
                "success": False,
                "roll": 0, "all_rolls": [], "total": 0, "dc": 0,
                "damage": 0, "damage_rolls": [],
                "critical": None,
                "narration_hint": intent.get("reason", "Действие противоречит законам мира."),
                "state_delta": {},
            }

        # Диалог и исследование без проверок — бросок не нужен
        if action in ("dialogue", "exploration"):
            return {
                "success": True,
                "roll": 0, "all_rolls": [], "total": 0, "dc": 0,
                "damage": 0, "damage_rolls": [],
                "critical": None,
                "narration_hint": f"Действие типа {action} прошло без проверки.",
                "state_delta": {},
            }

        # === АТАКА ===
        if action == "attack":
            target_id = intent.get("target")
            target = world_state.get("entities", {}).get(target_id, {})
            target_ac = target.get("ac", intent.get("target_ac", 13))

            roll, all_rolls = DnDEngine.roll_d20(advantage, disadvantage)
            crit = None
            if roll == 20:
                crit = "hit"
            elif roll == 1:
                crit = "miss"

            total = roll + modifier
            hit = crit == "hit" or (crit != "miss" and total >= target_ac)

            damage = 0
            damage_rolls = []
            if hit:
                dice = intent.get("damage_dice", "1d6")
                damage, damage_rolls = DnDEngine.roll_dice(dice)
                if crit == "hit":
                    # 5e критический урон: бросаем игровые кубики ещё раз и складываем
                    extra, extra_rolls = DnDEngine.roll_dice(re.sub(r"[+-]\s*\d+", "", dice))
                    damage += extra
                    damage_rolls += extra_rolls

            state_delta = {}
            if hit and target_id:
                state_delta = {"hp": {target_id: -damage}}

            return {
                "success": hit,
                "roll": roll, "all_rolls": all_rolls,
                "total": total, "dc": target_ac,
                "damage": damage, "damage_rolls": damage_rolls,
                "critical": crit,
                "narration_hint": (
                    f"Атака по {target_id or 'цели'}: бросок {roll}+{modifier}={total} vs AC {target_ac}. "
                    f"{'Критическое попадание!' if crit == 'hit' else 'Критический промах!' if crit == 'miss' else 'Попадание.' if hit else 'Промах.'}"
                    + (f" Урон: {damage}." if hit else "")
                ),
                "state_delta": state_delta,
            }

        # === SKILL CHECK ===
        if action == "skill_check":
            skill = (intent.get("skill") or "").lower()
            difficulty = intent.get("difficulty", "medium")
            dc = intent.get("dc") or DEFAULT_DC.get(difficulty, 15)

            roll, all_rolls = DnDEngine.roll_d20(advantage, disadvantage)
            total = roll + modifier
            success = total >= dc

            return {
                "success": success,
                "roll": roll, "all_rolls": all_rolls,
                "total": total, "dc": dc,
                "damage": 0, "damage_rolls": [],
                "critical": "hit" if roll == 20 else ("miss" if roll == 1 else None),
                "narration_hint": (
                    f"Проверка {skill or 'навыка'} (DC {dc}): {roll}+{modifier}={total}. "
                    f"{'Успех.' if success else 'Провал.'}"
                ),
                "state_delta": {},
            }

        # === SPELL ===
        if action == "spell":
            spell_name = intent.get("spell", "?")
            # Простая модель: если у заклинания есть атака — катим атаку, иначе skill_check на спасбросок
            if intent.get("damage_dice"):
                # Spell attack — как обычная атака
                roll, all_rolls = DnDEngine.roll_d20(advantage, disadvantage)
                target_ac = intent.get("target_ac", 13)
                total = roll + modifier
                hit = roll != 1 and (roll == 20 or total >= target_ac)
                damage, damage_rolls = (DnDEngine.roll_dice(intent["damage_dice"]) if hit else (0, []))
                return {
                    "success": hit,
                    "roll": roll, "all_rolls": all_rolls,
                    "total": total, "dc": target_ac,
                    "damage": damage, "damage_rolls": damage_rolls,
                    "critical": "hit" if roll == 20 else ("miss" if roll == 1 else None),
                    "narration_hint": f"Заклинание {spell_name}: {roll}+{modifier}={total} vs AC {target_ac}. {'Попадание.' if hit else 'Промах.'} Урон: {damage}.",
                    "state_delta": {"hp": {intent["target"]: -damage}} if hit and intent.get("target") else {},
                }
            # Иначе — спасбросок цели против DC заклинания
            spell_dc = intent.get("dc", 13)
            save_mod = intent.get("save_modifier", 0)
            save_roll, all_rolls = DnDEngine.roll_d20()
            saved = (save_roll + save_mod) >= spell_dc
            damage = 0
            damage_rolls = []
            if intent.get("damage_dice"):
                damage, damage_rolls = DnDEngine.roll_dice(intent["damage_dice"])
                if saved:
                    damage //= 2  # 5e: половина урона при удачном спасброске
            return {
                "success": not saved,
                "roll": save_roll, "all_rolls": all_rolls,
                "total": save_roll + save_mod, "dc": spell_dc,
                "damage": damage, "damage_rolls": damage_rolls,
                "critical": None,
                "narration_hint": f"Заклинание {spell_name}: цель {'выдерживает' if saved else 'не выдерживает'} спасбросок (DC {spell_dc}). Урон: {damage}.",
                "state_delta": {"hp": {intent["target"]: -damage}} if intent.get("target") else {},
            }

        # Неизвестный action — fallback на skill_check medium
        roll, all_rolls = DnDEngine.roll_d20()
        total = roll + modifier
        dc = DEFAULT_DC["medium"]
        return {
            "success": total >= dc,
            "roll": roll, "all_rolls": all_rolls,
            "total": total, "dc": dc,
            "damage": 0, "damage_rolls": [],
            "critical": None,
            "narration_hint": f"Неопределённое действие, бросок d20={roll}+{modifier}={total} vs DC {dc}.",
            "state_delta": {},
        }

    # === LEGACY API (старый App.tsx-фронт всё ещё его дёргает через /logic) ===
    @staticmethod
    def process_action(action_text: str) -> dict:
        """
        Сохранён для обратной совместимости со старым фронтом.
        Новый код должен звать resolve() с уже распарсенным intent от Судьи.
        """
        roll, _ = DnDEngine.roll_d20()
        action_lower = action_text.lower()
        is_attack = any(word in action_lower for word in ['атаку', 'атака', 'бью', 'удар', 'выстрел', 'убиваю', 'рублю'])
        is_mental = any(word in action_lower for word in ['понять', 'вспомнить', 'подумать', 'осознать', 'проверить', 'осмотреться', 'гляжу', 'замечаю'])

        if is_attack:
            result = "Успех" if roll >= 12 else "Провал"
            damage, _ = DnDEngine.roll_dice("1d8+3") if result == "Успех" else (0, [])
            return {
                "roll": roll, "result": result, "damage": damage,
                "text": f"Бросок атаки: {roll}. {'Попадание!' if result == 'Успех' else 'Промах.'} Урон: {damage}."
            }
        elif is_mental:
            result = "Успех" if roll >= 6 else "Неудача"
            return {
                "roll": roll, "result": result,
                "text": f"Проверка внимания/памяти: {roll}. {'Вас посещает озарение или вы замечаете деталь.' if result == 'Успех' else 'Вам не удается ничего вспомнить или заметить.'}"
            }
        else:
            result = "Успех" if roll >= 10 else "Провал"
            return {
                "roll": roll, "result": result,
                "text": f"Сложность действия пройдена. {'Результат благоприятный.' if result == 'Успех' else 'Действие не принесло желаемого итога.'}"
            }
