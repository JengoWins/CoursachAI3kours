import json
import random
import datetime
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
import tensorflow as tf

# Загрузка данных
with open('./DataSets/dialog_data.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

dataset = dataset[:5000]

# Специальные токены
START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'

# Подготовка данных
inputs, outputs = [], []
for item in dataset:
    dialog = item['dialog']
    for i in range(len(dialog) - 1):
        if dialog[i]['speaker'] == 'user':
            inputs.append(dialog[i]['text'])
            outputs.append(f"{START_TOKEN} {dialog[i + 1]['text']} {END_TOKEN}")

# Токенизация
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>', filters='')
tokenizer.fit_on_texts(inputs + outputs)

# Добавляем специальные токены
tokenizer.word_index[START_TOKEN] = len(tokenizer.word_index) + 1
tokenizer.word_index[END_TOKEN] = len(tokenizer.word_index) + 1
tokenizer.word_index[PAD_TOKEN] = 0
tokenizer.index_word[0] = PAD_TOKEN

vocab_size = len(tokenizer.word_index) + 1

# Преобразование входов
X = tokenizer.texts_to_sequences(inputs)
X = pad_sequences(X, maxlen=50, padding='post', truncating='post', value=tokenizer.word_index[PAD_TOKEN])

# Преобразование выходов
y = tokenizer.texts_to_sequences(outputs)
y = pad_sequences(y, maxlen=50, padding='post', truncating='post', value=tokenizer.word_index[PAD_TOKEN])

# Создаем модель с правильной архитектурой
input_layer = Input(shape=(50,))
embedding = Embedding(vocab_size, 256, mask_zero=True)(input_layer)
lstm = LSTM(512, return_sequences=True)(embedding)  # Возвращаем все последовательности
output_layer = Dense(vocab_size, activation='softmax')(lstm)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=1, batch_size=128)


randIndex = random.randint(1, 9999)
datefolder = datetime.date.today()
path = f'./Models/Model_{datefolder}_{randIndex}'
os.mkdir(path)
model.save(f'{path}/dialog_model.keras')
with open(f'{path}/tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())


# Функция генерации ответа
def generate_response_JSON(input_text, model=model, tokenizer=tokenizer, max_len=50, temp=0.4):
    # Токенизация входа
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post',
                              truncating='post', value=tokenizer.word_index[PAD_TOKEN])

    # Инициализация выхода
    output_seq = [tokenizer.word_index[START_TOKEN]]

    for _ in range(max_len):
        # Подготовка текущего состояния
        seq = pad_sequences([output_seq], maxlen=max_len, padding='post',
                            value=tokenizer.word_index[PAD_TOKEN])

        # Предсказание следующего токена
        pred = model.predict(input_seq, verbose=0)[0, len(output_seq) - 1, :]

        # Применение температуры
        pred = np.log(pred) / temp
        exp_pred = np.exp(pred)
        pred = exp_pred / np.sum(exp_pred)

        # Выбор токена
        next_token = np.random.choice(len(pred), p=pred)

        if next_token == tokenizer.word_index[END_TOKEN]:
            break

        output_seq.append(next_token)

    # Преобразование в текст
    response = ' '.join([tokenizer.index_word[t] for t in output_seq if t not in
                         [tokenizer.word_index[START_TOKEN], tokenizer.word_index[PAD_TOKEN]]])
    return response