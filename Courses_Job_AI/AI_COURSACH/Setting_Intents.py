#!/usr/bin/env python
# coding: utf-8
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import Setting_JSON
from AI_COURSACH import Setting_TXT
from DataSets.dataset1 import BOT_CONFIG_BASE

X_text = []  # ['Хэй', 'хаюхай', 'Хаюшки', ...]
y = []  # ['hello', 'hello', 'hello', ...]

for intent, intent_data in BOT_CONFIG_BASE['intents'].items():
    for example in intent_data['examples']:
        X_text.append(example)
        y.append(intent)


vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
X = vectorizer.fit_transform(X_text)
clf = LinearSVC()
clf.fit(X, y)


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
        if example and distance / len(example) <= 0.4:
            return intent


#Значит тут получаем реплику, которая была найдена в датасете и выводим случайный ответ из датасета
def get_answer_by_intent(intent):
    print(intent)
    if intent in BOT_CONFIG_BASE['intents']:
        responses = BOT_CONFIG_BASE['intents'][intent]['responses']
        if responses:
            return random.choice(responses)


def get_failure_phrase():
    failure_phrases = BOT_CONFIG_BASE['failure_phrases']
    return random.choice(failure_phrases)


stats = {'intentBASE': 0, 'generateJSON': 0, 'generateTXT': 0, 'failure': 0}

dialog_history = []

context = ''


def botDialog(replica):
    #Попытка создать историю
    global context
    dialog_history.append(replica)
    #context = ' '.join(dialog_history[-3:]) if len(dialog_history) > 1 else replica
    print("History dialog: ", dialog_history)

    #Сначала просчитываем базовые вещи
    intent = classify_intent(replica)
    if intent:
        answer = get_answer_by_intent(intent)
        print("Какой результат предсказания? : ", answer)
        if answer:
            stats['intentBASE'] += 1
            return answer


    #Либо ищем, что можно найти в JSON
    answerJSON = Setting_JSON.get_response(replica)
    if answerJSON and answerJSON.lower() not in ['я не знаю', 'не понимаю']:
        stats['generateJSON'] += 1
        return answerJSON

    #На крайняк из txt
    answerTXT = Setting_TXT.predict_response(replica)
    if answerTXT and answerTXT.lower() not in ['я не знаю', 'не понимаю']:
        stats['generateTXT'] += 1
        return answerTXT


botDialog('Сколько времени?')
