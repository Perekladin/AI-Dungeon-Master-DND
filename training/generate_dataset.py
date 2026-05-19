import json

# Эталонные примеры для обучения агента "Судья" (Logic Agent)
# Формат: input (ввод), thought (логика CoT), result (JSON ответ)

golden_examples = [
    # --- БОЕВЫЕ СЦЕНАРИИ ---
    {
        "input": "Атака скелета мечом. Бросок d20: 14. Бонус атаки: +5. КД (AC) скелета: 13. Урон: 1d8+3 (выпало 5+3).",
        "thought": "1. Определяем целевой КД (AC): 13. 2. Рассчитываем итоговый бросок: 14 (база) + 5 (бонус) = 19. 3. Проверка: 19 >= 13. 4. Результат: Попадание. 5. Урон: 5 + 3 = 8.",
        "result": "{\"success\": true, \"roll\": 14, \"bonus\": 5, \"total\": 19, \"target_ac\": 13, \"damage\": 8, \"effect\": \"Hit with longsword\"}"
    },
    {
        "input": "Атака гоблина луком. Бросок d20: 4. Бонус атаки: +4. КД (AC) гоблина: 15.",
        "thought": "1. Определяем КД цели: 15. 2. Итоговый бросок: 4 + 4 = 8. 3. Проверка: 8 < 15. 4. Результат: Промах.",
        "result": "{\"success\": false, \"roll\": 4, \"bonus\": 4, \"total\": 8, \"target_ac\": 15, \"damage\": 0, \"effect\": \"Missed with shortbow\"}"
    },
    {
        "input": "Критический провал! Атака орка. Бросок d20: 1. Бонус: +6. КД орка: 16.",
        "thought": "1. Выпала натуральная 1. В D&D 5e это автоматический провал атаки вне зависимости от модификаторов. 2. Результат: Критический промах.",
        "result": "{\"success\": false, \"roll\": 1, \"bonus\": 6, \"total\": 7, \"target_ac\": 16, \"damage\": 0, \"effect\": \"Critical miss\"}"
    },
    
    # --- ПРОВЕРКИ НАВЫКОВ (SKILL CHECKS) ---
    {
        "input": "Попытка взломать замок. СУ (DC): 15. Ловкость (Воровские инструменты): 1d20 (выпало 12) + 4 бонус.",
        "thought": "1. Определяем сложность (DC): 15. 2. Сумма броска: 12 + 4 = 16. 3. Проверка: 16 >= 15. 4. Результат: Успех. Замок открыт.",
        "result": "{\"success\": true, \"roll\": 12, \"bonus\": 4, \"total\": 16, \"target_dc\": 15, \"damage\": 0, \"effect\": \"Lock picked successfully\"}"
    },
    {
        "input": "Прыжок через расщелину (Атлетика). СУ (DC): 12. Бросок: 7. Бонус: +2.",
        "thought": "1. Цель DC: 12. 2. Итого: 7 + 2 = 9. 3. Проверка: 9 < 12. 4. Результат: Провал. Игрок не допрыгнул.",
        "result": "{\"success\": false, \"roll\": 7, \"bonus\": 2, \"total\": 9, \"target_dc\": 12, \"damage\": 5, \"effect\": \"Fell into the chasm, took fall damage\"}"
    }
]

def save_dataset(filename="training/judge_data.jsonl"):
    with open(filename, 'w', encoding='utf-8') as f:
        for ex in golden_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"Dataset saved to {filename} with {len(golden_examples)} examples.")

if __name__ == "__main__":
    save_dataset()
