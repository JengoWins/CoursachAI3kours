#!/usr/bin/env python
# coding: utf-8
import datetime
import random

import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, BatchNormalization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from DataSets.dataset1 import BOT_CONFIG_BASE


X_text = []  # ['Хэй', 'хаюхай', 'Хаюшки', ...]
y = []  # ['hello', 'hello', 'hello', ...]

for intent, intent_data in BOT_CONFIG_BASE['intents'].items():
    for example in intent_data['examples']:
        X_text.append(example)
        y.append(intent)

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
    for example in BOT_CONFIG_BASE['intents'][intent]['examples']:
        example = clear_phrase(example)
        distance = nltk.edit_distance(replica, example)
        if example and distance / len(example) <= 0.6:
            return intent


#Значит тут получаем реплику, которая была найдена в датасете и выводим случайный ответ из датасета
def get_answer_by_intent(intent):
    if intent in BOT_CONFIG_BASE['intents']:
        responses = BOT_CONFIG_BASE['intents'][intent]['responses']
        if responses:
            return random.choice(responses)


"""
with open('Dialogs/dialogues.txt', encoding='utf-8') as f:
    content = f.read()

dialogues_str = content.split('\n\n')
dialogues = [dialogue_str.split('\n')[:2] for dialogue_str in dialogues_str]
dialogues_filtered = []
questions = set()

for dialogue in dialogues:
    if len(dialogue) != 2:
        continue
    question, answer = dialogue
    question = clear_phrase(question[2:])
    answer = answer[2:]
    if question != '' and question not in questions:
        questions.add(question)
        dialogues_filtered.append([question, answer])

dialogues_structured = {}  # {'word': [['...word...', 'answer'], ...], ...}

for question, answer in dialogues_filtered:
    words = set(question.split(' '))
    for word in words:
        if word not in dialogues_structured:
            dialogues_structured[word] = []
        dialogues_structured[word].append([question, answer])

print(dialogues_structured)
dialogues_structured_cut = {}

for word, pairs in dialogues_structured.items():
    pairs.sort(key=lambda pair: len(pair[0]))
    dialogues_structured_cut[word] = pairs[:1000]

#print(dialogues_structured_cut)
# replica -> word1, word2, word3, ... -> dialogues_structured[word1] + dialogues_structured[word2] + ... -> mini_dataset


def generate_answer(replica):
    replica = clear_phrase(replica)
    words = set(replica.split(' '))
    mini_dataset = []
    for word in words:
        if word in dialogues_structured_cut:
            mini_dataset += dialogues_structured_cut[word]

    # TODO убрать повторы из mini_dataset

    answers = []  # [[distance_weighted, question, answer]]

    for question, answer in mini_dataset:
        if abs(len(replica) - len(question)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            distance_weighted = distance / len(question)
            if distance_weighted < 0.2:
                answers.append([distance_weighted, question, answer])

    if answers:
        return min(answers, key=lambda three: three[0])[2]
"""

with open('Dialogs/dialogues.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

dialogues = []
current_dialogue = []
for line in lines:
    line = line.strip()
    if line.startswith('-'):
        line = line[1:].strip()
        if line:  # Игнорировать пустые строки
            current_dialogue.append(line)
    else:
        if current_dialogue:
            dialogues.append(current_dialogue)
            current_dialogue = []

dialogues = dialogues[:5000]
# Добавляем последний диалог, если файл не заканчивается пустой строкой
if current_dialogue:
    dialogues.append(current_dialogue)

pairs = []
for dialogue in dialogues:
    for i in range(len(dialogue) - 1):
        pairs.append((dialogue[i], dialogue[i + 1]))

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#print(tf.config.list_physical_devices('GPU'))

pairs = pairs[:1000]
input_texts = [item[0] for item in dialogues]
output_texts = [item[1] for item in dialogues]

# Векторизация с общим токенизатором
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + output_texts)
vocab_size = len(tokenizer.word_index) + 1

# Преобразование входных данных
input_sequences = tokenizer.texts_to_sequences(input_texts)
input_data = pad_sequences(input_sequences)
max_length = input_data.shape[1]  # Запоминаем максимальную длину

# Преобразование выходных данных
output_sequences = tokenizer.texts_to_sequences(output_texts)
output_data = pad_sequences(output_sequences, maxlen=max_length)

# Преобразуем в one-hot encoding
output_data = tf.keras.utils.to_categorical(output_data, num_classes=vocab_size)

# Проверка размеров
print(f"Input shape: {input_data.shape}")  # (n_samples, max_length)
print(f"Output shape: {output_data.shape}") # (n_samples, max_length, vocab_size)

# Создаем модель
model = tf.keras.Sequential([
    Embedding(vocab_size, 1024, input_length=max_length),
    Bidirectional(LSTM(512, return_sequences=True, kernel_regularizer=l2(0.0001))),  # Возвращаем последовательность
    Dropout(0.5),
    BatchNormalization(),
    Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0001))),
    Dropout(0.5),
    BatchNormalization(),  # Нормализует активации
    Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.0001))),
    Dropout(0.5),
    BatchNormalization(),  # Нормализует активации
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0001))),
    Dropout(0.5),
    BatchNormalization(),
    Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.0001))),
    Dropout(0.5),
    BatchNormalization(),
    Bidirectional(LSTM(10, return_sequences=True, kernel_regularizer=l2(0.0001))),
    Dropout(0.5),
    BatchNormalization(),
    TimeDistributed(Dense(vocab_size, activation='softmax'))
])
#optimizer='adam'
#optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),  # Остановка при переобучении
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),  # Авто-настройка LR
    tf.keras.callbacks.ModelCheckpoint(f'E:/TensorflowModel/my_model_stopped_{datetime.date.today()}.keras', save_best_only=True)
]

model.fit(input_data, output_data, epochs=50, batch_size=16, callbacks=callbacks)
randIndex = random.randint(1,9999)
model.save(f'E:/TensorflowModel/my_model_{randIndex}_{datetime.date.today()}.keras')


def generate_human_like_answer(prompt, temperature=0.7, max_length=30, max_output_words=50):
    try:
        # 1. Проверка и предобработка ввода
        if not isinstance(prompt, str) or not prompt.strip():
            return ""

        prompt = prompt.strip()

        # 2. Токенизация и паддинг входа
        input_seq = tokenizer.texts_to_sequences([prompt])
        if not input_seq or len(input_seq[0]) == 0:
            return ""

        input_seq = pad_sequences(input_seq, maxlen=max_length, padding='post')
        generated_sequence = input_seq[0].tolist()

        # 3. Генерация ответа по словам
        for _ in range(max_output_words):
            # Подготовка текущей последовательности
            current_seq = pad_sequences([generated_sequence], maxlen=max_length, padding='post')

            # Получаем предсказания модели
            preds = model.predict(current_seq, verbose=0)[0]

            # Для рекуррентных моделей берем последний временной шаг
            if preds.ndim > 1:
                preds = preds[-1]

            # Применяем температурное преобразование
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)

            # Выбираем следующее слово (не детерминировано)
            next_word_id = np.random.choice(len(preds), p=preds)

            # Проверка на конец предложения
            if (next_word_id == tokenizer.word_index.get('<end>') or
                    next_word_id == tokenizer.word_index.get('<eos>') or
                    len(generated_sequence) >= max_output_words):
                break

            generated_sequence.append(next_word_id)

        # 4. Преобразуем индексы в текст
        result = tokenizer.sequences_to_texts([generated_sequence])[0]

        # 5. Постобработка
        result = result.replace(prompt, "").strip()
        result = result.capitalize()

        # Удаляем частичные предложения после последней точки
        if '.' in result:
            result = result[:result.rfind('.') + 1]

        return result if result else "Я не знаю, что ответить."

    except Exception as e:
        print(f"Generation error: {str(e)}")
        return "Извините, произошла ошибка."


def get_failure_phrase():
    failure_phrases = BOT_CONFIG_BASE['failure_phrases']
    return random.choice(failure_phrases)


stats = {'intent': 0, 'generate': 0, 'failure': 0}


def botDialog(replica):
    # NLU
    intent = classify_intent(replica)
    # Answer generation
    # выбор заготовленной реплики
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats['intent'] += 1
            return answer
    # вызов генеративной модели
    answer = generate_human_like_answer(replica)
    if answer:
        stats['generate'] += 1
        return answer

    # берем заглушку
    stats['failure'] += 1
    return get_failure_phrase()


botDialog('Сколько времени?')
