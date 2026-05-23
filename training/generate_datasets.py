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
        # Длинные фразы, которые игроки реально пишут в чате
        "Я вхожу в таверну и оглядываюсь.",
        "Захожу в комнату, осматриваюсь по сторонам.",
        "Иду по коридору, прислушиваясь к звукам.",
        "Подхожу ближе к {place} и внимательно смотрю.",
        "Оглядываюсь по сторонам, ищу что-нибудь подозрительное.",
        "Делаю шаг вперёд и осматриваю помещение.",
        "Прохожу в зал и оглядываю посетителей.",
        "Спускаюсь по лестнице вниз.",
        "Открываю дверь и заглядываю внутрь.",
        "Сажусь за стол в углу.",
        "Иду к стойке тавернщика.",
        "Подхожу к окну, смотрю наружу.",
        "Беру факел со стены.",
        "Поднимаю фонарь и иду дальше.",
        "Жду, пока кто-нибудь подойдёт.",
        "Стою и слушаю разговоры вокруг.",
        "Иду в сторону {place}.",
        "Возвращаюсь обратно тем же путём.",
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
        "Удар застаёт {monster}а врасплох. Кость трещит, кровь брызжет тёмной струёй. Глаза твари мутнеют от боли. Эхо удара затихает в холодном воздухе подземелья.",
        "Сталь находит щель в кольчуге. {monster} издаёт булькающий стон, отшатываясь к каменной стене. Под ногами расплывается лужа крови. Запах железа смешивается с потом и страхом.",
        "Лезвие проходит сквозь жёсткую кожу твари. {monster} ревёт от боли, отшатываясь и обнажая клыки. Светильник на стене дёргается от взмаха.",
    ],
    "miss": [
        "Удар проходит вскользь. {monster} с рычанием отступает в тень за бочками, готовясь к ответному выпаду. Где-то капает вода с потолка. Рукоять скользкая от пота.",
        "Клинок рассекает только воздух. {monster_cap} оскаливается, обнажая жёлтые зубы. Сердце колотится как молот по наковальне. Холод ползёт по спине под кольчугой.",
        "Промах. {monster} уворачивается с неожиданной ловкостью. Дыхание сбилось, факельный свет дрожит на влажных стенах. Бой не окончен.",
        "Сталь высекает искры о камень — удар уходит мимо. {monster} рычит и делает шаг ближе, обнажая когти.",
    ],
    "skill_success": [
        "Замок поддаётся с тихим щелчком, и тяжёлая дубовая створка приоткрывается на палец. Из щели тянет холодом и запахом сырости. Где-то в глубине шевелится что-то живое.",
        "Тёмный коридор открывает свои секреты — каменная плита в углу подаётся, обнажая узкий проход. Пыль веков оседает на плечи, факел потрескивает.",
        "Получилось. Половицы под ногами не скрипнули — стража у двери продолжает дремать, опершись на алебарду.",
        "Узор на каменной плите складывается в осмысленный знак — древнюю руну. Память о прочитанных свитках всплывает сама собой.",
    ],
    "skill_fail": [
        "Замок не поддаётся, что-то внутри застряло намертво. С лестницы доносится скрип половиц — кто-то идёт с фонарём.",
        "Пальцы соскальзывают, отмычка ломается с тихим хрустом. Время уходит, во дворе лают собаки.",
        "Половица скрипит предательски громко. В соседней комнате слышится недовольное бормотание просыпающегося стражника.",
        "Камень не подаётся под рукой. Откуда-то сверху доносится шорох — будто кто-то прислушивается.",
    ],
    "spell_hit": [
        "Заклинание срывается с пальцев, и {monster} обволакивает пламя. Кожа твари трескается, обугливаясь по краям. Жар откатывает волной к лицу.",
        "Магия вспыхивает в воздухе. {monster} кричит — звук, который останется в памяти. Воздух пахнет грозой и палёной плотью.",
        "Ледяной луч пронзает {monster}а насквозь. Иней покрывает каменный пол вокруг. Где-то наверху падает капля с сосульки.",
    ],
    "impossible": [
        "Воздух тяжелеет. Что-то в самой ткани мира отказывается подчиняться вашему намерению. Ветер шелестит листвой за стенами таверны.",
        "Намерение угасает, не успев оформиться. Этот мир не позволяет такого — здесь правят сталь и молитвы, а не своеволие смертных.",
        "Боги, чьи имена высечены на алтарях, явно не слышат вашей просьбы. Камень остаётся камнем, а тело — телом.",
    ],
    "dialogue": [
        "Тавернщик медленно вытирает оловянную кружку, обдумывая ваши слова. Его взгляд скользит по рукам, по поясу с оружием. — «Барняра? Знал такого. Был тут на прошлой неделе, расплатился серебром и ушёл с какой-то ведьмой».",
        "В зале на миг повисает тишина — слышно лишь треск дров в очаге. Старик у стойки оборачивается. — «О таких вещах не спрашивают вслух, путник. Особенно в этих стенах».",
        "Стражник оглядывается через плечо, понижает голос. — «Барняр? Иди вниз по реке, до старой мельницы. Только не говори, что я тебе сказал — иначе мою голову вздёрнут на воротах».",
        "Торговец прищуривается, поглаживая бороду. — «Знание стоит денег, незнакомец. Кошель открой — тогда поговорим».",
        "Жрец медленно кивает, перебирая чётки из кости. — «Всё в руках Светоносного. Что предначертано — то и будет».",
    ],
    "exploration": [
        "Зал таверны тонет в дымном полумраке. Пахнет жареным мясом, элем и потом. У очага дремлет старый пёс, в углу шестеро мужиков режутся в кости на медяки. Тавернщик за стойкой провожает вас взглядом.",
        "Узкий каменный коридор тянется в темноту. Запах плесени становится сильнее с каждым шагом. Эхо подошв возвращается странно искажённым, будто стены здесь не совсем мёртвые.",
        "Помещение тонет в полумраке. Холод пробирает сквозь подошвы — каменный пол сырой и липкий от чего-то тёмного. В дальнем углу поблёскивает металл — то ли монеты, то ли клинок.",
        "Лес стоит мёртвый. Ни птиц, ни ветра — лишь скрип старых сосен и далёкий вой волка. Иней лежит на корнях, хотя осень ещё не сдала своих прав.",
        "Двор замка пуст. Гербовые знамёна обвисли в безветрии, по брусчатке плетётся бродячая курица. Над сторожевой башней кружит ворон.",
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
# 6. LOGIC — объединённый датасет для одного агента, играющего три роли
# ===========================================================================
# В каждой строке есть system-prompt — модель учится различать роли по нему.
# Старые train_orchestrator/judge/keeper больше не нужны.

SYS_CLASSIFY = (
    "Ты классификатор намерений игрока в D&D 5e. "
    "Ответь ОДНИМ словом из набора: АТАКА, НАВЫК, ЗАКЛИНАНИЕ, ДИАЛОГ, ИССЛЕДОВАНИЕ, НЕВОЗМОЖНО. "
    "Никаких объяснений, никакого другого текста."
)
SYS_PARSE = (
    "Ты — Судья D&D 5e. Твоя ЕДИНСТВЕННАЯ задача: распарсить русскую фразу игрока в строгий JSON.\n"
    "НЕ принимай решений об успехе/провале — это сделает движок. НЕ катай кубики — это сделает движок.\n"
    "Верни ровно один JSON-объект и ничего больше. Схема:\n"
    '{"action":"attack"|"skill_check"|"spell"|"dialogue"|"exploration"|"impossible",'
    '"target":<строка или null>,"skill":<строка или null>,"weapon":<строка или null>,'
    '"spell":<строка или null>,"modifier":<целое>,"advantage":<bool>,"disadvantage":<bool>,'
    '"difficulty":"easy"|"medium"|"hard"|null,"damage_dice":<строка типа "1d8+3" или null>,'
    '"reason":<строка>}\n'
    "Если действие физически невозможно — action='impossible' и заполни reason."
)
SYS_STATE = (
    "Ты — Хранитель состояния мира D&D. На вход получаешь лог события. "
    "Верни СТРОГИЙ JSON с обновлением мира и ничего больше. Схема:\n"
    '{"location":<строка или null>,"summary":<строка ≤200 символов>,'
    '"enemies_alive":[<id>],"hp_changes":{<entity_id>:<delta>},"flags":{<ключ>:<значение>}}\n'
    "summary — краткая выжимка сцены для следующего хода. Только факты."
)


# === Чарник + NPC реестр — источник истины для whitelist ===
# Эти данные подменяют общие WEAPONS/MONSTERS из верхней части файла.
# Logic-агент учится только на ID которые РЕАЛЬНО есть в нашем pipeline.

def _load_character_sheet():
    """Читаем character_sheet.json чтобы вытащить актуальный whitelist."""
    sheet_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dnd_core_models", "character_sheet.json"
    )
    if not os.path.exists(sheet_path):
        return None
    with open(sheet_path, encoding="utf-8") as f:
        return json.load(f)


_CHAR = _load_character_sheet()
# Если запускаемся в Colab без чарника — используем дефолты следопыта-3.
if _CHAR:
    CHAR_WEAPONS = _CHAR["weapons"]
    CHAR_SPELLS = _CHAR["spells"]
    CHAR_SKILLS_TRAINED = [n for n, s in _CHAR["skills"].items() if s.get("proficient")]
    CHAR_SKILLS_ALL = list(_CHAR["skills"].keys())
else:
    CHAR_WEAPONS = [
        {"id": "longsword", "name": "длинный меч", "attack_bonus": 4, "damage_dice": "1d8+2"},
        {"id": "shortbow",  "name": "короткий лук", "attack_bonus": 5, "damage_dice": "1d6+3"},
        {"id": "dagger",    "name": "кинжал",      "attack_bonus": 5, "damage_dice": "1d4+3"},
    ]
    CHAR_SPELLS = [
        {"id": "hunters_mark", "name": "Метка охотника", "level": 1, "damage_dice": "1d6"},
        {"id": "cure_wounds",  "name": "Лечение ран",    "level": 1, "damage_dice": "1d8+3"},
    ]
    CHAR_SKILLS_TRAINED = ["athletics", "stealth", "perception", "survival", "intimidation", "nature"]
    CHAR_SKILLS_ALL = CHAR_SKILLS_TRAINED + ["acrobatics", "history", "investigation", "persuasion", "deception"]

# NPC типы которые умеет канонизировать registry
CANONICAL_NPCS = [
    ("bandit", "наёмник",   12, 11),
    ("guard",  "стражник",  16, 11),
    ("thief",  "вор",       13, 7),
    ("goblin", "гоблин",    15, 7),
    ("orc",    "орк",       13, 15),
    ("skeleton","скелет",   13, 13),
    ("wolf",   "волк",      13, 11),
]

# Не-наши оружия — для генерации negative-примеров (попытка → impossible)
FOREIGN_WEAPONS = [
    "shortsword", "greatsword", "rapier", "battleaxe", "warhammer", "spear",
    "crossbow", "greataxe", "scimitar", "javelin",
]
FOREIGN_SPELLS = [
    "fireball", "magic_missile", "lightning_bolt", "wish", "true_polymorph",
    "ray_of_frost", "sleep", "thunderwave", "shield",
]


def _gen_logic_classify(n: int) -> List[Dict]:
    """Классификация — обёртка над gen_orchestrator с SYS_CLASSIFY."""
    return [{"system": SYS_CLASSIFY, "user": ex["user"], "assistant": ex["assistant"]}
            for ex in gen_orchestrator(n)]


def _gen_logic_parse(n: int) -> List[Dict]:
    """Парсинг — переписан под whitelist чарника и canonical NPC IDs."""
    out = []
    while len(out) < n:
        kind = random.choices(
            ["attack_known", "attack_foreign", "spell_known", "spell_foreign",
             "skill_check", "dialogue", "exploration", "impossible_action"],
            weights=[30, 15, 10, 10, 15, 8, 7, 5]
        )[0]

        location = random.choice(LOCATIONS)
        npc_id, npc_ru, npc_ac, _ = random.choice(CANONICAL_NPCS)

        if kind == "attack_known":
            wpn = random.choice(CHAR_WEAPONS)
            atk_verb = random.choice(["Атакую", "Бью", "Рублю", "Колю"])
            user = (f"Состояние мира: бой в {location} с {npc_ru} (AC {npc_ac}).\n"
                    f"Действие игрока: {atk_verb} {npc_ru}а {wpn['name']}ом.")
            assistant = {
                "action": "attack",
                "target": npc_id,  # КАНОНИЧЕСКИЙ id — bandit, guard, ...
                "skill": None,
                "weapon": wpn["id"],  # ID из чарника
                "spell": None,
                "modifier": wpn["attack_bonus"],
                "advantage": False, "disadvantage": False,
                "difficulty": None,
                "damage_dice": wpn["damage_dice"],
                "reason": f"Атака {wpn['name']}ом по {npc_ru}.",
            }
        elif kind == "attack_foreign":
            # Игрок пытается использовать оружие которого у него нет → impossible
            foreign = random.choice(FOREIGN_WEAPONS)
            foreign_ru = {"shortsword": "короткий меч", "greatsword": "двуручный меч",
                          "rapier": "рапира", "battleaxe": "боевой топор",
                          "warhammer": "молот", "spear": "копьё",
                          "crossbow": "арбалет", "greataxe": "двуручный топор",
                          "scimitar": "ятаган", "javelin": "метательное копьё"}.get(foreign, foreign)
            user = (f"Состояние мира: {location} с {npc_ru}.\n"
                    f"Действие игрока: Атакую {npc_ru}а {foreign_ru}ом.")
            assistant = {
                "action": "impossible",
                "target": npc_id, "skill": None,
                "weapon": None, "spell": None,
                "modifier": 0, "advantage": False, "disadvantage": False,
                "difficulty": None, "damage_dice": None,
                "reason": f"У персонажа нет такого оружия как {foreign_ru}. Доступно только: длинный меч, короткий лук, кинжал.",
            }
        elif kind == "spell_known":
            spell = random.choice(CHAR_SPELLS)
            user = (f"Состояние мира: бой в {location} с {npc_ru}.\n"
                    f"Действие игрока: Кастую {spell['name']} в {npc_ru}а.")
            assistant = {
                "action": "spell",
                "target": npc_id, "skill": None,
                "weapon": None, "spell": spell["id"],
                "modifier": 4, "advantage": False, "disadvantage": False,
                "difficulty": None, "damage_dice": spell.get("damage_dice"),
                "reason": f"Заклинание {spell['name']} по {npc_ru}.",
            }
        elif kind == "spell_foreign":
            foreign = random.choice(FOREIGN_SPELLS)
            foreign_ru = {"fireball": "Огненный шар", "magic_missile": "Магическая стрела",
                          "lightning_bolt": "Молния", "wish": "Исполнение желаний",
                          "ray_of_frost": "Луч холода", "sleep": "Сон",
                          "thunderwave": "Громовая волна", "shield": "Щит",
                          "true_polymorph": "Истинное перевоплощение"}.get(foreign, foreign)
            user = (f"Состояние мира: {location}.\n"
                    f"Действие игрока: Кастую {foreign_ru}.")
            assistant = {
                "action": "impossible",
                "target": None, "skill": None,
                "weapon": None, "spell": None,
                "modifier": 0, "advantage": False, "disadvantage": False,
                "difficulty": None, "damage_dice": None,
                "reason": f"Это заклинание ({foreign_ru}) не входит в репертуар персонажа. Доступно: Метка охотника, Лечение ран.",
            }
        elif kind == "skill_check":
            sk = random.choice(CHAR_SKILLS_ALL)
            actions_map = {
                "athletics":     "перелезть через стену",
                "stealth":       "прокрасться мимо стражи",
                "perception":    "осмотреть комнату внимательно",
                "survival":      "найти следы в лесу",
                "intimidation":  "запугать тавернщика",
                "nature":        "вспомнить про этот вид деревьев",
                "acrobatics":    "сделать сальто через стол",
                "history":       "вспомнить о роде Барняра",
                "investigation": "обыскать тело",
                "persuasion":    "уговорить торговца снизить цену",
                "deception":     "соврать стражнику что я гонец",
            }
            action_text = actions_map.get(sk, f"проверить {sk}")
            difficulty = random.choice(["easy", "medium", "hard"])
            user = (f"Состояние мира: {location}.\n"
                    f"Действие игрока: Хочу {action_text}.")
            assistant = {
                "action": "skill_check",
                "target": None, "skill": sk,
                "weapon": None, "spell": None,
                "modifier": 3, "advantage": False, "disadvantage": False,
                "difficulty": difficulty, "damage_dice": None,
                "reason": f"Проверка навыка {sk}.",
            }
        elif kind == "dialogue":
            phrases = ["Где Барняр?", "Сколько за комнату?", "Кто здесь главный?",
                       "Покажи свои товары.", "Я ищу проводника на север."]
            user = (f"Состояние мира: {location}.\n"
                    f"Действие игрока: Я говорю: '{random.choice(phrases)}'")
            assistant = {
                "action": "dialogue",
                "target": "tavernkeeper" if "таверн" in location.lower() else None,
                "skill": None, "weapon": None, "spell": None,
                "modifier": 0, "advantage": False, "disadvantage": False,
                "difficulty": None, "damage_dice": None,
                "reason": "Прямой диалог.",
            }
        elif kind == "exploration":
            verbs = ["Осматриваюсь.", "Иду на север.", "Захожу в дверь.",
                     "Поднимаюсь по лестнице.", "Подхожу к окну.",
                     "Я захожу в таверну и оглядываюсь."]
            user = (f"Состояние мира: {location}.\nДействие игрока: {random.choice(verbs)}")
            assistant = {
                "action": "exploration",
                "target": None, "skill": None,
                "weapon": None, "spell": None,
                "modifier": 0, "advantage": False, "disadvantage": False,
                "difficulty": None, "damage_dice": None,
                "reason": "Перемещение или осмотр.",
            }
        else:  # impossible_action
            phrase = random.choice(IMPOSSIBLE_ACTIONS)
            user = (f"Состояние мира: {location}.\nДействие игрока: {phrase}")
            assistant = {
                "action": "impossible",
                "target": None, "skill": None,
                "weapon": None, "spell": None,
                "modifier": 0, "advantage": False, "disadvantage": False,
                "difficulty": None, "damage_dice": None,
                "reason": "Действие нарушает законы мира или возможности персонажа.",
            }

        out.append({"system": SYS_PARSE, "user": user,
                    "assistant": json.dumps(assistant, ensure_ascii=False)})
    return out


def _gen_logic_state(n: int) -> List[Dict]:
    """State delta — переиспользуем gen_keeper, но с canonical NPC."""
    out = []
    for ex in gen_keeper(n):
        out.append({"system": SYS_STATE, "user": ex["user"], "assistant": ex["assistant"]})
    return out


def gen_logic(count: int) -> List[Dict]:
    """Смешанный датасет для единого Logic-агента (classify + parse + state).

    Распределение: 35% classify (часто вызывается), 50% parse (главная задача),
    15% state (более простая задача). Парс делает большую часть тяжёлой работы
    Logic'а, потому что от него зависит intent → весь pipeline.
    """
    classify_count = int(count * 0.35)
    parse_count = int(count * 0.5)
    state_count = count - classify_count - parse_count

    out = []
    out += _gen_logic_classify(classify_count)
    out += _gen_logic_parse(parse_count)
    out += _gen_logic_state(state_count)

    random.shuffle(out)
    return out


# ===========================================================================
# DISPATCH
# ===========================================================================

GENERATORS = {
    "orchestrator": gen_orchestrator,
    "judge":        gen_judge,
    "keeper":       gen_keeper,
    "tactician":    gen_tactician,
    "storyteller":  gen_storyteller,
    "logic":        gen_logic,
}

DEFAULTS = {
    "orchestrator": 800,
    "judge":        1500,
    "keeper":       1000,
    "tactician":    600,
    "storyteller":  1200,
    "logic":        3000,  # три задачи в одном — нужно больше примеров
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
