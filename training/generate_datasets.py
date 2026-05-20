"""
Генератор синтетических обучающих датасетов для всех 5 агентов D&D Core.

Создаёт {agent}_data.jsonl рядом с собой. Формат каждой строки:
    {"user": "...", "assistant": "..."}

Запуск:
    python generate_datasets.py              # все 5 агентов
    python generate_datasets.py --agent judge --count 1500
    python generate_datasets.py --golden     # дописать ручные эталонные примеры из golden_examples.py

Идея: 80% синтетики из шаблонов + 20% «золотых» вручную размеченных примеров.
Синтетика покрывает БАЗУ (распознавание паттернов), golden — корнер-кейсы.
"""

import argparse
import json
import random
import os
from typing import List, Dict

random.seed(42)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# СПРАВОЧНИКИ
# ===========================================================================

WEAPONS = [
    ("меч",         "1d8+3",  "longsword"),
    ("кинжал",      "1d4+2",  "dagger"),
    ("лук",         "1d6+2",  "shortbow"),
    ("топор",       "1d10+3", "battleaxe"),
    ("копьё",       "1d6+3",  "spear"),
    ("молот",       "1d8+3",  "warhammer"),
    ("арбалет",     "1d8+2",  "crossbow"),
    ("посох",       "1d6+1",  "quarterstaff"),
]

MONSTERS = [
    ("гоблин",   "goblin",     15),
    ("скелет",   "skeleton",   13),
    ("орк",      "orc",        13),
    ("волк",     "wolf",       12),
    ("зомби",    "zombie",     8),
    ("кобольд",  "kobold",     12),
    ("огр",      "ogre",       11),
    ("бандит",   "bandit",     12),
    ("крыса",    "giant_rat",  12),
    ("паук",     "giant_spider", 14),
]

SKILLS = [
    ("атлетика",         "athletics",       "str"),
    ("акробатика",       "acrobatics",      "dex"),
    ("скрытность",       "stealth",         "dex"),
    ("ловкость рук",     "sleight_of_hand", "dex"),
    ("внимательность",   "perception",      "wis"),
    ("выживание",        "survival",        "wis"),
    ("убеждение",        "persuasion",      "cha"),
    ("запугивание",      "intimidation",    "cha"),
    ("обман",            "deception",       "cha"),
    ("история",          "history",         "int"),
    ("магия",            "arcana",          "int"),
    ("анализ",           "investigation",   "int"),
]

SPELLS = [
    ("огненный шар",       "fireball",       "8d6",   "dex"),
    ("волшебная стрела",   "magic_missile",  "1d4+1", None),
    ("огненная стрела",    "firebolt",       "1d10",  None),
    ("луч холода",         "ray_of_frost",   "1d8",   None),
    ("сон",                "sleep",          None,    "wis"),
    ("свет",               "light",          None,    None),
    ("лечащее слово",      "healing_word",   "1d4+3", None),
]

IMPOSSIBLE_ACTIONS = [
    "Я взмахиваю руками и взлетаю в небо.",
    "Я телепортируюсь в королевский замок.",
    "Я создаю золото из воздуха силой воли.",
    "Я останавливаю время.",
    "Я призываю молнию голыми руками без заклинаний.",
    "Я воскрешаю павшего товарища, не имея способностей жреца.",
    "Я становлюсь невидимым по желанию.",
    "Я читаю мысли тавернщика, не будучи псионом.",
    "Я разламываю каменную стену кулаком.",
    "Я плыву по лаве не получая урона.",
]

LOCATIONS = [
    "сырая таверна 'Хромой Дракон'", "тёмный лес", "развалины храма",
    "канализация под городом", "пещера гоблинов", "болото у дороги",
    "горный перевал", "кладбище за деревней", "башня волшебника",
    "подземелье склепа", "торговая площадь", "берег чёрной реки",
]


# ===========================================================================
# 1. ORCHESTRATOR
# ===========================================================================

def gen_orchestrator(count: int) -> List[Dict]:
    examples = []

    attack_templates = [
        "Атакую {target} {weapon}ом.",
        "Бью {target} {weapon}ом.",
        "Стреляю в {target} из {weapon}а.",
        "Рублю {target} {weapon}ом наотмашь.",
        "Я хочу убить {target}.",
        "Кидаю {weapon} в {target}а.",
    ]
    skill_templates = [
        "Пытаюсь {action} ({skill}).",
        "Делаю проверку {skill} чтобы {action}.",
        "Прячусь за {place}.",
        "Прыгаю через {place}.",
        "Взламываю замок на двери.",
        "Подслушиваю разговор за стеной.",
        "Уговариваю стражника пропустить меня.",
        "Запугиваю тавернщика.",
    ]
    spell_templates = [
        "Кастую {spell} в {target}а.",
        "Произношу заклинание {spell}.",
        "Атакую {spell}ом.",
        "Лечу себя заклинанием.",
    ]
    dialogue_templates = [
        'Я говорю: "{phrase}"',
        'Спрашиваю: "{phrase}"',
        'Кричу: "{phrase}"',
        'Шепчу спутнику: "{phrase}"',
    ]
    explore_templates = [
        "Осматриваюсь.",
        "Иду на север.",
        "Захожу в дверь.",
        "Изучаю комнату.",
        "Подхожу к {place}.",
        "Открываю сундук.",
        "Жду что будет.",
        "Изучаю карту.",
    ]

    phrases = ["Где Барняр?", "Сколько стоит эль?", "Откуда вы?", "Кто здесь главный?", "Помогите мне"]
    places = ["камнем", "колодцем", "столом", "обрывом", "ямой", "статуей"]
    actions = ["перелезть", "залезть", "проскользнуть", "найти", "вспомнить"]

    while len(examples) < count:
        bucket = random.choices(
            ["АТАКА", "НАВЫК", "ЗАКЛИНАНИЕ", "ДИАЛОГ", "ИССЛЕДОВАНИЕ", "НЕВОЗМОЖНО"],
            weights=[25, 25, 15, 15, 15, 5]
        )[0]
        if bucket == "АТАКА":
            t = random.choice(attack_templates)
            mon = random.choice(MONSTERS)[0]
            wpn = random.choice(WEAPONS)[0]
            user = t.format(target=mon, weapon=wpn)
        elif bucket == "НАВЫК":
            t = random.choice(skill_templates)
            sk = random.choice(SKILLS)[0]
            user = t.format(skill=sk, action=random.choice(actions), place=random.choice(places))
        elif bucket == "ЗАКЛИНАНИЕ":
            t = random.choice(spell_templates)
            sp = random.choice(SPELLS)[0]
            mon = random.choice(MONSTERS)[0]
            user = t.format(spell=sp, target=mon)
        elif bucket == "ДИАЛОГ":
            t = random.choice(dialogue_templates)
            user = t.format(phrase=random.choice(phrases))
        elif bucket == "ИССЛЕДОВАНИЕ":
            t = random.choice(explore_templates)
            user = t.format(place=random.choice(places))
        else:  # НЕВОЗМОЖНО
            user = random.choice(IMPOSSIBLE_ACTIONS)
        examples.append({"user": user, "assistant": bucket})
    return examples


# ===========================================================================
# 2. JUDGE
# ===========================================================================

def gen_judge(count: int) -> List[Dict]:
    examples = []
    while len(examples) < count:
        kind = random.choices(
            ["attack", "skill_check", "spell", "dialogue", "exploration", "impossible"],
            weights=[30, 30, 15, 10, 10, 5]
        )[0]

        if kind == "attack":
            mon_ru, mon_id, mon_ac = random.choice(MONSTERS)
            wpn_ru, dmg, wpn_id = random.choice(WEAPONS)
            mod = random.randint(2, 6)
            adv = random.random() < 0.1
            disadv = random.random() < 0.1 and not adv
            user = f"Состояние мира: бой в {random.choice(LOCATIONS)} с {mon_ru} (AC {mon_ac}).\nДействие игрока: Атакую {mon_ru} {wpn_ru}ом."
            assistant = {
                "action": "attack",
                "target": mon_id,
                "skill": None, "weapon": wpn_id, "spell": None,
                "modifier": mod, "advantage": adv, "disadvantage": disadv,
                "difficulty": None, "damage_dice": dmg,
                "reason": f"Боевая атака {wpn_ru}ом по {mon_ru}.",
            }
        elif kind == "skill_check":
            sk_ru, sk_id, _ = random.choice(SKILLS)
            mod = random.randint(0, 5)
            difficulty = random.choice(["easy", "medium", "hard"])
            actions = [f"проскользнуть мимо стражи ({sk_ru})", f"запугать тавернщика", f"вспомнить руны",
                       f"перелезть через стену", f"взломать замок", f"уговорить торговца"]
            user = f"Состояние мира: {random.choice(LOCATIONS)}.\nДействие игрока: {random.choice(actions)}."
            assistant = {
                "action": "skill_check",
                "target": None, "skill": sk_id, "weapon": None, "spell": None,
                "modifier": mod, "advantage": False, "disadvantage": False,
                "difficulty": difficulty, "damage_dice": None,
                "reason": f"Проверка навыка {sk_ru} сложности {difficulty}.",
            }
        elif kind == "spell":
            sp_ru, sp_id, dmg, save = random.choice(SPELLS)
            mon_ru, mon_id, mon_ac = random.choice(MONSTERS)
            mod = random.randint(3, 6)
            user = f"Состояние мира: бой в {random.choice(LOCATIONS)}.\nДействие игрока: Кастую {sp_ru} в {mon_ru}а."
            assistant = {
                "action": "spell",
                "target": mon_id, "skill": None, "weapon": None, "spell": sp_id,
                "modifier": mod, "advantage": False, "disadvantage": False,
                "difficulty": None, "damage_dice": dmg,
                "reason": f"Атакующее заклинание {sp_ru}.",
            }
        elif kind == "dialogue":
            phrases = ["Где Барняр?", "Я ищу проводника", "Сколько за комнату?", "Покажи свои товары"]
            user = f"Состояние мира: {random.choice(LOCATIONS)}.\nДействие игрока: Я говорю: '{random.choice(phrases)}'"
            assistant = {
                "action": "dialogue",
                "target": None, "skill": None, "weapon": None, "spell": None,
                "modifier": 0, "advantage": False, "disadvantage": False,
                "difficulty": None, "damage_dice": None,
                "reason": "Чистый диалог без проверки.",
            }
        elif kind == "exploration":
            verbs = ["Осматриваюсь.", "Иду на север.", "Захожу в дверь.", "Открываю сундук."]
            user = f"Состояние мира: {random.choice(LOCATIONS)}.\nДействие игрока: {random.choice(verbs)}"
            assistant = {
                "action": "exploration",
                "target": None, "skill": None, "weapon": None, "spell": None,
                "modifier": 0, "advantage": False, "disadvantage": False,
                "difficulty": None, "damage_dice": None,
                "reason": "Перемещение или осмотр сцены.",
            }
        else:  # impossible
            phrase = random.choice(IMPOSSIBLE_ACTIONS)
            user = f"Состояние мира: {random.choice(LOCATIONS)}.\nДействие игрока: {phrase}"
            assistant = {
                "action": "impossible",
                "target": None, "skill": None, "weapon": None, "spell": None,
                "modifier": 0, "advantage": False, "disadvantage": False,
                "difficulty": None, "damage_dice": None,
                "reason": "Действие нарушает физику или возможности персонажа.",
            }

        examples.append({"user": user, "assistant": json.dumps(assistant, ensure_ascii=False)})
    return examples


# ===========================================================================
# 3. KEEPER
# ===========================================================================

def gen_keeper(count: int) -> List[Dict]:
    examples = []
    while len(examples) < count:
        loc = random.choice(LOCATIONS)
        mon_ru, mon_id, _ = random.choice(MONSTERS)

        event_type = random.choices(
            ["combat_hit", "combat_miss", "monster_killed", "skill_success", "skill_fail", "exploration", "dialogue"],
            weights=[20, 15, 15, 15, 10, 15, 10]
        )[0]

        if event_type == "combat_hit":
            dmg = random.randint(3, 12)
            user = (f"Текущая локация: {loc}\nИгрок: Атакую {mon_ru}.\n"
                    f"Произошло: Клинок находит цель, {mon_ru} ранен.\n"
                    f"Урон/изменения: {{\"hp\": {{\"{mon_id}\": -{dmg}}}}}\n"
                    f"Верни JSON-обновление.")
            assistant = {
                "location": loc, "summary": f"Бой с {mon_ru} в {loc[:40]}, противник ранен.",
                "enemies_alive": [mon_id], "hp_changes": {mon_id: -dmg}, "flags": {"in_combat": True},
            }
        elif event_type == "combat_miss":
            user = (f"Текущая локация: {loc}\nИгрок: Атакую {mon_ru}.\n"
                    f"Произошло: Удар уходит мимо, {mon_ru} увернулся.\n"
                    f"Урон/изменения: {{}}\nВерни JSON-обновление.")
            assistant = {
                "location": loc, "summary": f"Бой с {mon_ru} продолжается, последний удар мимо.",
                "enemies_alive": [mon_id], "hp_changes": {}, "flags": {"in_combat": True},
            }
        elif event_type == "monster_killed":
            user = (f"Текущая локация: {loc}\nИгрок: Добиваю {mon_ru}.\n"
                    f"Произошло: {mon_ru.capitalize()} падает замертво.\n"
                    f"Урон/изменения: {{\"hp\": {{\"{mon_id}\": -20}}}}\nВерни JSON-обновление.")
            assistant = {
                "location": loc, "summary": f"{mon_ru.capitalize()} мёртв. Бой завершён.",
                "enemies_alive": [], "hp_changes": {mon_id: -20}, "flags": {"in_combat": False},
            }
        elif event_type == "skill_success":
            sk = random.choice(SKILLS)[0]
            user = (f"Текущая локация: {loc}\nИгрок: Делаю проверку {sk}.\n"
                    f"Произошло: Получилось — игрок добивается результата.\n"
                    f"Урон/изменения: {{}}\nВерни JSON-обновление.")
            assistant = {
                "location": loc, "summary": f"Проверка {sk} в {loc[:40]} прошла успешно.",
                "enemies_alive": [], "hp_changes": {}, "flags": {f"checked_{sk}": True},
            }
        elif event_type == "skill_fail":
            sk = random.choice(SKILLS)[0]
            user = (f"Текущая локация: {loc}\nИгрок: Делаю проверку {sk}.\n"
                    f"Произошло: Не вышло.\nУрон/изменения: {{}}\nВерни JSON-обновление.")
            assistant = {
                "location": loc, "summary": f"Проверка {sk} провалена.",
                "enemies_alive": [], "hp_changes": {}, "flags": {f"failed_{sk}": True},
            }
        elif event_type == "exploration":
            new_loc = random.choice(LOCATIONS)
            user = (f"Текущая локация: {loc}\nИгрок: Иду дальше.\n"
                    f"Произошло: Игрок перемещается в новую область.\n"
                    f"Урон/изменения: {{}}\nВерни JSON-обновление.")
            assistant = {
                "location": new_loc, "summary": f"Игрок переместился из {loc[:30]} в {new_loc[:30]}.",
                "enemies_alive": [], "hp_changes": {}, "flags": {},
            }
        else:  # dialogue
            user = (f"Текущая локация: {loc}\nИгрок: Говорит с NPC.\n"
                    f"Произошло: NPC отвечает на вопросы.\nУрон/изменения: {{}}\nВерни JSON-обновление.")
            assistant = {
                "location": loc, "summary": f"Разговор с NPC в {loc[:40]}.",
                "enemies_alive": [], "hp_changes": {}, "flags": {"talked_to_npc": True},
            }

        examples.append({"user": user, "assistant": json.dumps(assistant, ensure_ascii=False)})
    return examples


# ===========================================================================
# 4. TACTICIAN
# ===========================================================================

def gen_tactician(count: int) -> List[Dict]:
    examples = []
    while len(examples) < count:
        mon_ru, mon_id, _ = random.choice(MONSTERS)
        tag = random.choice(["Mindless", "Cunning", "Cowardly", "Aggressive"])
        hp_cur = random.randint(1, 30)
        hp_max = 30
        ratio = hp_cur / hp_max

        scenario = (
            f"Бой. Монстры: {{\"{mon_id}\": {{\"name\": \"{mon_ru}\", \"hp_current\": {hp_cur}, "
            f"\"hp_max\": {hp_max}, \"behavior_tag\": \"{tag}\"}}}}. "
            f"Игрок только что: attack → {'успех' if random.random() < 0.5 else 'провал'}."
        )

        if tag == "Cowardly" and ratio < 0.3:
            plan = {"monster_id": mon_id, "action": "отступает в укрытие, пытается сбежать", "target": "none"}
        elif tag == "Mindless":
            plan = {"monster_id": mon_id, "action": "тупо бросается в ближнюю атаку", "target": "player"}
        elif tag == "Cunning":
            plan = {"monster_id": mon_id, "action": "обходит игрока сбоку для атаки в спину", "target": "player"}
        else:  # Aggressive
            plan = {"monster_id": mon_id, "action": "наносит сильный удар оружием", "target": "player"}

        examples.append({"user": scenario, "assistant": json.dumps(plan, ensure_ascii=False)})
    return examples


# ===========================================================================
# 5. STORYTELLER
# ===========================================================================

STORYTELLER_TEMPLATES = {
    "hit": [
        "Клинок входит в плоть с глухим хрустом. {monster} оседает, хрипя кровью сквозь стиснутые зубы. В воздухе разливается металлический запах раны. Тень от факела дрожит на каменной стене.",
        "Удар застаёт {monster}а врасплох. Кость трещит, кровь брызжет тёмной струёй. Вы видите как глаза твари мутнеют от боли. Эхо удара затихает в холодном воздухе.",
        "Сталь находит щель в доспехе. {monster} издаёт булькающий стон, отшатываясь назад. Под ногами начинает расплываться лужа крови. Запах железа смешивается с потом и страхом.",
    ],
    "miss": [
        "Удар проходит вскользь. {monster} с рычанием отступает в тень, готовясь к ответному выпаду. Где-то капает вода. Ваша рукоять скользкая от пота.",
        "Клинок рассекает только воздух. {monster_cap} оскаливается, обнажая жёлтые зубы. Сердце колотится как молот по наковальне. Холод ползёт по спине.",
        "Промах. {monster} уворачивается с неожиданной для его размеров ловкостью. Ваше дыхание сбилось. Бой не окончен.",
    ],
    "skill_success": [
        "Получается. Замок поддаётся с тихим щелчком, и створка приоткрывается на палец. Из щели тянет холодом и запахом сырости. Где-то в глубине что-то шевелится.",
        "Удаётся. Тёмный коридор открывает свои секреты — стена в углу подаётся, обнажая узкий проход. Пыль оседает на плечи.",
        "Получилось. Тавернщик кивает и понижает голос: 'Барняр был здесь два дня назад. Спрашивал про карты.' Его глаза бегают по залу.",
    ],
    "skill_fail": [
        "Не выходит. Замок не поддаётся, что-то внутри застряло намертво. С лестницы доносится скрип половиц — кто-то идёт.",
        "Провал. Ваши пальцы соскальзывают, отмычка ломается с тихим хрустом. Время уходит. Где-то лает собака.",
        "Не получилось. Тавернщик мрачно качает головой: 'Я ничего не слышал, господин. Не моё дело.' Разговор окончен.",
    ],
    "spell_hit": [
        "Заклинание срывается с пальцев, и {monster} обволакивает пламя. Кожа твари трескается, обугливаясь по краям. Жар откатывает волной к вам в лицо.",
        "Магия вспыхивает в воздухе. {monster} кричит — звук, который останется в памяти. Воздух пахнет озоном и палёной плотью.",
    ],
    "impossible": [
        "Воздух тяжелеет. Что-то в самой ткани реальности отказывается подчиняться вашему намерению. Ничего не происходит, лишь ветер шелестит листьями.",
        "Намерение ваше угасает, не успев оформиться. Этот мир не позволяет такого.",
    ],
    "dialogue": [
        "Тавернщик медленно вытирает кружку, обдумывая ваши слова. Его взгляд скользит по вашим рукам, по поясу с оружием.",
        "В зале на миг повисает тишина. Чей-то кашель в углу. Затем разговор возобновляется — но уже тише.",
    ],
    "exploration": [
        "Помещение тонет в полумраке. Холод пробирает сквозь подошвы — каменный пол сырой и липкий. В дальнем углу что-то поблёскивает.",
        "Узкий коридор тянется в темноту. Запах плесени становится сильнее с каждым шагом. Эхо ваших шагов возвращается странно искажённым.",
    ],
}


def gen_storyteller(count: int) -> List[Dict]:
    examples = []
    while len(examples) < count:
        kind = random.choices(
            ["hit", "miss", "skill_success", "skill_fail", "spell_hit", "impossible", "dialogue", "exploration"],
            weights=[20, 15, 15, 10, 10, 5, 10, 15]
        )[0]
        loc = random.choice(LOCATIONS)
        mon = random.choice(MONSTERS)[0]
        tpl = random.choice(STORYTELLER_TEMPLATES[kind])
        narration = tpl.format(monster=mon, monster_cap=mon.capitalize())

        # Сборка user-инпута в формате, который сервер реально подаёт
        if kind in ("hit", "spell_hit"):
            hint = f"Атака успешна, {mon} ранен."
            action = f"Атакую {mon}а."
        elif kind == "miss":
            hint = f"Атака не достигла {mon}а."
            action = f"Атакую {mon}а."
        elif kind == "skill_success":
            hint = "Проверка навыка успешна."
            action = "Пытаюсь взломать замок."
        elif kind == "skill_fail":
            hint = "Проверка навыка провалена."
            action = "Пытаюсь взломать замок."
        elif kind == "impossible":
            hint = "Действие противоречит законам мира."
            action = random.choice(IMPOSSIBLE_ACTIONS)
        elif kind == "dialogue":
            hint = "Диалог без проверки."
            action = 'Я говорю: "Где Барняр?"'
        else:  # exploration
            hint = "Перемещение без проверки."
            action = "Осматриваюсь."

        user = (
            f"Локация: {loc}.\n"
            f"Действие игрока: {action}\n"
            f"Исход механики (для контекста, НЕ цитируй цифры): {hint}.\n"
            f"Опиши сцену атмосферно."
        )
        examples.append({"user": user, "assistant": narration})
    return examples


# ===========================================================================
# DISPATCH
# ===========================================================================

GENERATORS = {
    "orchestrator": gen_orchestrator,
    "judge":        gen_judge,
    "keeper":       gen_keeper,
    "tactician":    gen_tactician,
    "storyteller":  gen_storyteller,
}

DEFAULTS = {
    "orchestrator": 800,
    "judge":        1500,
    "keeper":       1000,
    "tactician":    600,
    "storyteller":  1200,
}


def save_jsonl(records: List[Dict], filename: str):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[+] {path}  ({len(records)} записей)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=list(GENERATORS) + ["all"], default="all")
    p.add_argument("--count", type=int, default=None, help="Сколько примеров (default — по умолчанию агента)")
    args = p.parse_args()

    targets = list(GENERATORS) if args.agent == "all" else [args.agent]
    for agent in targets:
        n = args.count if args.count else DEFAULTS[agent]
        records = GENERATORS[agent](n)
        save_jsonl(records, f"{agent}_data.jsonl")

    print("\nДатасеты готовы. Теперь запускай train_<agent>.py в Colab.")


if __name__ == "__main__":
    main()
