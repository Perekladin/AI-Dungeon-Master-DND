import random
import re

class DnDEngine:
    @staticmethod
    def roll_d20():
        return random.randint(1, 20)

    @staticmethod
    def roll_dice(dice_str):
        # Format: 1d8+3
        match = re.match(r"(\d+)d(\d+)([+-]\d+)?", dice_str)
        if not match:
            return 8 # Default
        num, sides, mod = match.groups()
        total = sum(random.randint(1, int(sides)) for _ in range(int(num)))
        if mod:
            total += int(mod)
        return total

    @staticmethod
    def process_action(action_text):
        roll = DnDEngine.roll_d20()
        action_lower = action_text.lower()
        
        is_attack = any(word in action_lower for word in ['атаку', 'бью', 'удар', 'выстрел', 'убиваю'])
        is_mental = any(word in action_lower for word in ['понять', 'вспомнить', 'подумать', 'осознать', 'проверить', 'осмотреться', 'гляжу'])
        
        if is_attack:
            result = "Успех" if roll >= 12 else "Провал"
            damage = DnDEngine.roll_dice("1d8+3") if result == "Успех" else 0
            return {
                "roll": roll,
                "result": result,
                "damage": damage,
                "text": f"Бросок атаки: {roll}. {'Попадание!' if result == 'Успех' else 'Промах.'} Урон: {damage}."
            }
        elif is_mental:
            result = "Успех" if roll >= 6 else "Неудача"
            return {
                "roll": roll,
                "result": result,
                "text": f"Проверка внимания/памяти: {roll}. {'Вас посещает озарение или вы замечаете деталь.' if result == 'Успех' else 'Вам не удается ничего вспомнить или заметить.'}"
            }
        else:
            result = "Успех" if roll >= 10 else "Провал"
            return {
                "roll": roll,
                "result": result,
                "text": f"Сложность действия пройдена. {'Результат благоприятный.' if result == 'Успех' else 'Действие не принесло желаемого итога.'}"
            }
