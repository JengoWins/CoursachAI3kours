import nltk


def clear_phrase(phrase):
    phrase = phrase.lower()
    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    result = ''.join(symbol for symbol in phrase if symbol in alphabet)
    return result.strip()


with open('Dialogs/dialogues.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()


dialogues = []
current_dialogue = []
for line in lines:
    line = line.strip()
    if line.startswith('-'):
        line = line[1:].strip()
        if line:
            current_dialogue.append(line)
    else:
        if current_dialogue:
            dialogues.append(current_dialogue)
            current_dialogue = []

if current_dialogue:
    dialogues.append(current_dialogue)


txt_pairs = []
for dialogue in dialogues:
    for i in range(len(dialogue) - 1):
        txt_pairs.append((dialogue[i], dialogue[i + 1]))


def generate_response_TXT(replica):
    replica = clear_phrase(replica)
    replica_words = set(replica.split())

    candidates = []
    for question, answer in txt_pairs:
        question_cleaned = clear_phrase(question)
        question_words = set(question_cleaned.split())

        common_words = replica_words & question_words
        if not common_words:
            continue

        if abs(len(replica) - len(question_cleaned)) / len(question_cleaned) >= 0.2:
            continue

        distance = nltk.edit_distance(replica, question_cleaned)
        distance_weighted = distance / len(question_cleaned)
        if distance_weighted < 0.2:
            candidates.append((distance_weighted, question, answer))

    if candidates:
        return min(candidates, key=lambda x: x[0])[2]
    return None