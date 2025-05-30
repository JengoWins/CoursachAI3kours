#!/usr/bin/env python
# coding: utf-8
import random
import nltk
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from DataSets.dataset1 import BOT_CONFIG_BASE

X_text = []  # ['Хэй', 'хаюхай', 'Хаюшки', ...]
y = []  # ['hello', 'hello', 'hello', ...]

for intent, intent_data in BOT_CONFIG_BASE['intents'].items():
    for example in intent_data['examples']:
        X_text.append(example)
        y.append(intent)

print("Набор текста: ", X_text)
print("Набор ключей словарей: ", y)

#Он реально строит какуют экспертизу оценки через фит
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
X = vectorizer.fit_transform(X_text)
clf = LinearSVC()
clf.fit(X, y)


#Как-то очищает ненужные символы, пробелы всякие и т.д.
def clear_phrase(phrase):
    phrase = phrase.lower()
    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    result = ''.join(symbol for symbol in phrase if symbol in alphabet)
    return result.strip()


#Классифицирует наш запрос, находя в датасете нужный класс с репликами, а далее подбирает в датасете соотвествующий запрос
def classify_intent(replica):
    replica = clear_phrase(replica)
    intent = clf.predict(vectorizer.transform([replica]))[0]
    print(intent)
    for example in BOT_CONFIG_BASE['intents'][intent]['examples']:
        example = clear_phrase(example)
        distance = nltk.edit_distance(replica, example)
        if example and distance / len(example) <= 0.5:
            return intent


#Значит тут получаем реплику, которая была найдена в датасете и выводим случайный ответ из датасета
def get_answer_by_intent(intent):
    if intent in BOT_CONFIG_BASE['intents']:
        responses = BOT_CONFIG_BASE['intents'][intent]['responses']
        if responses:
            return random.choice(responses)


def generate_answer(replica):
    replica = clear_phrase(replica)
    #words = set(replica.split(' '))
    mini_dataset = []
    """
    for word in words:
        if word in dialogues_structured_cut:
            mini_dataset += dialogues_structured_cut[word]
    """
    # TODO убрать повторы из mini_dataset

    answers = []  # [[distance_weighted, question, answer]]
    print("Mini-Dataset : ", mini_dataset)
    for question, answer in mini_dataset:
        if abs(len(replica) - len(question)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            distance_weighted = distance / len(question)
            if distance_weighted < 0.2:
                answers.append([distance_weighted, question, answer])

    if answers:
        return min(answers, key=lambda three: three[0])[2]


#Если реплика не найдена, выводит случайную фразу и класса ошибок
def get_failure_phrase():
    failure_phrases = BOT_CONFIG_BASE['failure_phrases']
    return random.choice(failure_phrases)


stats = {'intent': 0, 'generate': 0, 'failure': 0}


def botDialog(replica):
    # NLU
    intent = classify_intent(replica)
    print("Классификация данных : ", intent)
    # Answer generation
    # выбор заготовленной реплики
    if intent:
        answer = get_answer_by_intent(intent)
        print("Чтоооо это? : ", answer)
        if answer:
            stats['intent'] += 1
            return answer

    # вызов генеративной модели
    answer = generate_answer(replica)
    print("Какая-то генеративная модель: ", answer)
    if answer:
        stats['generate'] += 1
        return answer

    # берем заглушку
    stats['failure'] += 1
    return get_failure_phrase()


botDialog('Сколько времени?')
