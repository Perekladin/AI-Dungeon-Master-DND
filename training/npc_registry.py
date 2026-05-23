"""
Реестр НПС — стат-блоки и алиасы.

Зачем: Logic-агент выдумывает разные target ID для одного и того же врага
(«наёмник» → bandit, потом → gnome_thief, потом → second_guerdian). Это
ломает state: нет связи между ходами, TacticalAI не находит врага в world_state.

Решение: централизованный справочник + нормализация имён.
"""

from typing import Optional


# Стат-блоки типовых НПС (на основе D&D 5e Monster Manual / стандартных стат-блоков)
NPC_TEMPLATES = {
    "bandit": {
        "name": "наёмник",
        "hp_max": 11, "hp_current": 11,
        "ac": 12,
        "behavior_tag": "Aggressive",
        "attack_bonus": 3,
        "damage_dice": "1d6+1",
    },
    "guard": {
        "name": "стражник",
        "hp_max": 11, "hp_current": 11,
        "ac": 16,
        "behavior_tag": "Defensive",
        "attack_bonus": 3,
        "damage_dice": "1d8+1",
    },
    "thief": {
        "name": "вор",
        "hp_max": 7, "hp_current": 7,
        "ac": 13,
        "behavior_tag": "Cunning",
        "attack_bonus": 4,
        "damage_dice": "1d4+2",
    },
    "goblin": {
        "name": "гоблин",
        "hp_max": 7, "hp_current": 7,
        "ac": 15,
        "behavior_tag": "Cowardly",
        "attack_bonus": 4,
        "damage_dice": "1d6+2",
    },
    "orc": {
        "name": "орк",
        "hp_max": 15, "hp_current": 15,
        "ac": 13,
        "behavior_tag": "Aggressive",
        "attack_bonus": 5,
        "damage_dice": "1d12+3",
    },
    "skeleton": {
        "name": "скелет",
        "hp_max": 13, "hp_current": 13,
        "ac": 13,
        "behavior_tag": "Mindless",
        "attack_bonus": 4,
        "damage_dice": "1d6+2",
    },
    "wolf": {
        "name": "волк",
        "hp_max": 11, "hp_current": 11,
        "ac": 13,
        "behavior_tag": "Aggressive",
        "attack_bonus": 4,
        "damage_dice": "2d4+2",
    },
    "rat": {
        "name": "гигантская крыса",
        "hp_max": 7, "hp_current": 7,
        "ac": 12,
        "behavior_tag": "Mindless",
        "attack_bonus": 4,
        "damage_dice": "1d4+2",
    },
    "tavernkeeper": {
        "name": "тавернщик",
        "hp_max": 9, "hp_current": 9,
        "ac": 10,
        "behavior_tag": "Cowardly",
        "attack_bonus": 1,
        "damage_dice": "1d4",
    },
    "merchant": {
        "name": "торговец",
        "hp_max": 9, "hp_current": 9,
        "ac": 10,
        "behavior_tag": "Cowardly",
        "attack_bonus": 1,
        "damage_dice": "1d4",
    },
}


# Алиасы для нормализации Logic-выходов и русских слов из user_input.
# Logic-агент выдаёт самые разные ID — приводим всё к каноническим типам выше.
NPC_ALIASES = {
    # Наёмники / бандиты — все одного типа
    "наёмник": "bandit", "наемник": "bandit",
    "бандит": "bandit", "разбойник": "bandit",
    "гном": "bandit", "gnome": "bandit",  # Logic часто пишет gnome для наёмника
    "gnome_thief": "bandit", "gnome_performer": "bandit",
    "townsmith": "bandit", "stranger": "bandit",
    "barneira": "bandit", "second_guerdian": "bandit",
    "запертый_бандит": "bandit",
    # Стражники
    "стражник": "guard", "страж": "guard",
    "охранник": "guard", "guardsman": "guard",
    # Воры
    "вор": "thief", "карманник": "thief", "грабитель": "thief",
    # Гоблины
    "гоблин": "goblin",
    # Орки
    "орк": "orc",
    # Скелеты
    "скелет": "skeleton", "костяк": "skeleton",
    # Волки и крысы
    "волк": "wolf", "крыса": "rat", "giant_rat": "rat",
    # Таверна
    "тавернщик": "tavernkeeper", "трактирщик": "tavernkeeper",
    # Торговцы
    "торговец": "merchant", "купец": "merchant", "лавочник": "merchant",
}


def canonicalize_target(target_raw: Optional[str], user_input: str = "") -> Optional[str]:
    """Приводит выдуманный Logic'ом target ID к каноническому ключу NPC_TEMPLATES.

    Логика поиска (по приоритету):
      1. target_raw в NPC_ALIASES → canonical id
      2. target_raw в NPC_TEMPLATES напрямую → возвращаем как есть
      3. Какое-то русское слово из user_input есть в NPC_ALIASES
      4. Возвращаем 'bandit' как дефолтного «противника» если в input есть боевые слова
      5. Возвращаем None
    """
    if not target_raw and not user_input:
        return None

    # 1. Прямое попадание в алиасы
    if target_raw:
        key = target_raw.lower().strip()
        if key in NPC_ALIASES:
            return NPC_ALIASES[key]
        if key in NPC_TEMPLATES:
            return key

    # 2. Поиск по user_input
    lo = (user_input or "").lower()
    for alias, canonical in NPC_ALIASES.items():
        if alias in lo:
            return canonical

    # 3. Если совсем непонятно но контекст боевой — дефолтный «наёмник»
    if target_raw:
        return "bandit"
    return None


def ensure_npc_spawned(world_state: dict, target_canonical: str) -> Optional[str]:
    """Гарантирует что target_canonical существует в world_state.entities.

    Возвращает реальный entity_id который теперь точно есть в state.
    Если такой type уже есть — нумерует (bandit_1, bandit_2 для второго наёмника).
    """
    if not target_canonical or target_canonical not in NPC_TEMPLATES:
        return None

    entities = world_state.setdefault("entities", {})

    # Если уже есть живой entity этого типа — используем его (продолжаем бой с тем же)
    for eid, e in entities.items():
        if isinstance(e, dict) and e.get("type") == target_canonical and e.get("hp_current", 0) > 0:
            return eid

    # Подбираем свободный ID
    base = target_canonical
    candidate = base
    n = 1
    while candidate in entities:
        n += 1
        candidate = f"{base}_{n}"

    # Клонируем шаблон
    template = NPC_TEMPLATES[target_canonical]
    entities[candidate] = {
        **template,
        "type": target_canonical,
        "spawned_turn": True,  # для лога — спавнили только что
    }
    return candidate


def name_for_entity(entity_id: str, world_state: dict) -> str:
    """Возвращает русское имя для entity_id (для нарратива)."""
    ent = world_state.get("entities", {}).get(entity_id, {})
    if ent and ent.get("name"):
        return ent["name"]
    # Fallback по типу
    typ = ent.get("type") if isinstance(ent, dict) else None
    if typ and typ in NPC_TEMPLATES:
        return NPC_TEMPLATES[typ]["name"]
    return entity_id
