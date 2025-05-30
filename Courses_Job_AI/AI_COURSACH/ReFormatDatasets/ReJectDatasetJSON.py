import json

# Ваш исходный набор данных
# Загрузка датасета из JSON-файла
with open('E:/Courses_Job_AI/AI_COURSACH/Dialogs/dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)


# Функция для разбора диалога
def parse_dialog(dialog_text):
    lines = dialog_text.split('\n')
    dialog = []
    for line in lines:
        if line.startswith('Собеседник: '):
            speaker = 'user'
            text = line.replace('Собеседник: ', '').strip()
        elif line.startswith('Ты: '):
            speaker = 'assistant'
            text = line.replace('Ты: ', '').strip()
        else:
            # Пропускаем строки, не относящиеся к диалогу (например, описание контекста)
            continue
        if text:  # Добавляем только непустые реплики
            dialog.append({'speaker': speaker, 'text': text})
    return dialog


# Обрабатываем все диалоги
processed_data = []
for item in dataset:
    # Разбираем input (историю диалога)
    dialog_history = parse_dialog(item['input'])

    # Добавляем output (последний ответ ассистента)
    dialog_history.append({
        'speaker': 'assistant',
        'text': item['output']
    })

    # Сохраняем результат
    processed_data.append({
        'name': item['name'],
        'dialog': dialog_history
    })

# Сохраняем в JSON файл
with open('dialog_data.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

print("Данные успешно сохранены в 'dialog_data.json'")