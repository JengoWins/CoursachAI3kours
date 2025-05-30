
with open('E:/Courses_Job_AI/AI_COURSACH/Dialogs/dialogues.txt', 'r', encoding='utf-8') as f:
    raw_data = f.readlines()

# Разбиваем на диалоги
dialogues = []
current_dialogue = []

for line in raw_data:
    if line.strip() == "" and current_dialogue:
        dialogues.append(current_dialogue)
        current_dialogue = []
    else:
        current_dialogue.append(line.strip())

if current_dialogue:
    dialogues.append(current_dialogue)

# Форматируем в нужный вид
formatted_dialogues = []

for dialogue in dialogues:
    formatted = []
    for i, line in enumerate(dialogue):
        speaker = "Человек" if i % 2 == 0 else "Бот"
        formatted.append(f"{speaker} - {line}")
    formatted_dialogues.append("\n".join(formatted))

# Сохраняем в файл
with open("formatted_dialogues.txt", "w", encoding="utf-8") as f:
    for dialogue in formatted_dialogues:
        f.write(dialogue + "\n\n")

print("Преобразованные диалоги сохранены в formatted_dialogues.txt")