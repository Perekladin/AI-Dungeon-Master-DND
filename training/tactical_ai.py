"""
Детерминированная тактика монстров — замена LLM-Tactician'a.

Принцип: тактика — это функция от behavior_tag, HP, состояния боя, позиции.
LLM здесь была overkill: правила укладываются в 100 строк Python и работают
надёжно, мгновенно и не галлюцинируют.

API:
    plan = TacticalAI.plan_turn(world_state, last_intent, mechanics)
    # plan == {"monster_id": str, "action": str, "target": str} or None
"""

import random
from typing import Optional


# behavior_tag → стратегия (порядок проверки матчится сверху вниз)
BEHAVIOR_RULES = {
    "Mindless":   "тупо бросается в ближайшего врага без хитростей",
    "Cunning":    "обходит цель сбоку, бьёт в спину или по слабому",
    "Cowardly":   "при HP<30% бежит, иначе атакует осторожно",
    "Aggressive": "атакует самого опасного с яростью",
    "Defensive":  "держит строй, защищает раненых союзников",
}


class TacticalAI:
    """Чистый функциональный модуль. Никакого состояния."""

    @staticmethod
    def plan_turn(world_state: dict, last_intent: dict, mechanics: dict) -> Optional[dict]:
        """
        Возвращает план хода для одного активного монстра или None если боя нет.

        Параметры:
            world_state.entities: dict {monster_id: {name, hp_current, hp_max, behavior_tag, ...}}
            last_intent.action: "attack"|"spell"|"skill_check"|...
            mechanics.success: bool — попал ли игрок в прошлом ходу
            mechanics.damage: int — нанесённый игроком урон

        Возвращает:
            {"monster_id": id, "action": str, "target": "player", "intent_kind": "attack"|"flee"|"defend"}
        """
        # Боя нет, если игрок не атаковал и нет активных монстров
        is_combat = last_intent.get("action") in ("attack", "spell")
        entities = world_state.get("entities", {}) or {}
        alive = {
            eid: e for eid, e in entities.items()
            if isinstance(e, dict) and e.get("hp_current", 1) > 0
        }
        if not alive and not is_combat:
            return None
        if not alive:
            return None

        # Выбираем монстра с приоритетом: 1) кого только что атаковали 2) Aggressive 3) случайный
        attacked_id = last_intent.get("target")
        if attacked_id and attacked_id in alive:
            chosen_id = attacked_id
        else:
            # Aggressive > Cunning > Mindless > Cowardly > Defensive
            priority = {"Aggressive": 4, "Cunning": 3, "Mindless": 2, "Cowardly": 1, "Defensive": 0}
            chosen_id = max(
                alive.keys(),
                key=lambda eid: priority.get(alive[eid].get("behavior_tag", "Mindless"), 2)
            )

        monster = alive[chosen_id]
        tag = monster.get("behavior_tag", "Mindless")
        name = monster.get("name", chosen_id)
        hp_ratio = monster.get("hp_current", 1) / max(monster.get("hp_max", 1), 1)

        # Применяем поведенческие правила
        if tag == "Cowardly" and hp_ratio < 0.3:
            return {
                "monster_id": chosen_id,
                "intent_kind": "flee",
                "action": f"{name} с раной в боку отступает к выходу, прикрывая голову",
                "target": "none",
            }

        if tag == "Defensive" and len(alive) >= 2:
            # Если есть раненый союзник — прикрывает его
            wounded = [eid for eid, e in alive.items()
                       if eid != chosen_id and e.get("hp_current", 99) < e.get("hp_max", 1) * 0.5]
            if wounded:
                return {
                    "monster_id": chosen_id,
                    "intent_kind": "defend",
                    "action": f"{name} закрывает собой раненого {alive[wounded[0]].get('name', wounded[0])}",
                    "target": "player",
                }

        # Боевые действия — атакуем игрока
        if tag == "Mindless":
            verbs = ["слепо бросается с разинутой пастью", "тупо машет когтями",
                    "ревёт и идёт напролом", "хлестко бьёт лапой"]
        elif tag == "Cunning":
            verbs = ["заходит со спины, целясь в шею", "выжидает удобный момент и колет в бок",
                    "имитирует выпад, потом бьёт по ногам", "обходит и атакует с фланга"]
        elif tag == "Aggressive":
            verbs = ["обрушивает шквал ударов", "рычит и кидается с яростью",
                    "не считаясь с защитой, рубит сплеча", "бросается в безудержную атаку"]
        elif tag == "Cowardly":
            verbs = ["осторожно тычет копьём с расстояния", "пытается ударить и сразу отпрянуть",
                    "кидает в вас камень и отступает на шаг"]
        else:  # Defensive
            verbs = ["обороняется, выжидая ошибку", "наносит точный сдерживающий удар"]

        return {
            "monster_id": chosen_id,
            "intent_kind": "attack",
            "action": f"{name} {random.choice(verbs)}",
            "target": "player",
        }


if __name__ == "__main__":
    # Quick test
    ws = {"entities": {
        "goblin_1": {"name": "гоблин", "hp_current": 5, "hp_max": 15, "behavior_tag": "Cunning"},
        "orc_1":    {"name": "орк",    "hp_current": 20, "hp_max": 25, "behavior_tag": "Aggressive"},
    }}
    intent = {"action": "attack", "target": "goblin_1"}
    mech = {"success": True, "damage": 4}
    print(TacticalAI.plan_turn(ws, intent, mech))
