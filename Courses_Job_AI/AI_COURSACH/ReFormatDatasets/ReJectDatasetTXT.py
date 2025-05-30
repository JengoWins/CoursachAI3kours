import json
from collections import defaultdict


def parse_dialogs(data):
    dialogs = []
    current_dialog = []

    for line in data:
        line = line.strip()
        if not line:
            if current_dialog:
                dialogs.append(current_dialog)
                current_dialog = []
            continue

        if ' - - ' in line:
            try:
                speaker, text = line.split(' - - ', 1)
                current_dialog.append((speaker.strip(), text.strip()))
            except ValueError:
                print(f"Ошибка разбора строки: {line}")
                continue

    if current_dialog:
        dialogs.append(current_dialog)

    return dialogs


def create_intent_dataset(dialogs):
    intent_data = defaultdict(lambda: {'patterns': [], 'responses': []})
    csv_data = []

    for dialog in dialogs:
        for i in range(0, len(dialog), 2):
            if i + 1 < len(dialog):
                user_text = dialog[i][1]
                bot_text = dialog[i + 1][1]

                # Улучшенное создание тега
                words = [w for w in user_text.split() if w.isalnum()]
                if not words:
                    continue

                tag = '_'.join(words[:3]).lower()  # Берем до 3 первых слов
                tag = ''.join(c for c in tag if c.isalnum() or c == '_')
                tag = tag[:30]  # Ограничиваем длину тега

                intent_data[tag]['patterns'].append(user_text)
                intent_data[tag]['responses'].append(bot_text)
                csv_data.append([user_text, tag])

    intents = [{'tag': k, 'patterns': v['patterns'], 'responses': v['responses']}
               for k, v in intent_data.items()]

    return {
        'intents': intents,
        'csv_data': csv_data
    }


def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    with open('formatted_dialogues.txt', 'r', encoding='utf-8') as f:
        dialog_data = f.readlines()

    parsed_dialogs = parse_dialogs(dialog_data)
    dataset = create_intent_dataset(parsed_dialogs)

    save_to_json(dataset['intents'], 'intents_dataset.json')

    print("Обработано диалогов:", len(parsed_dialogs))
    print("Создано интентов:", len(dataset['intents']))


if __name__ == '__main__':
    main()