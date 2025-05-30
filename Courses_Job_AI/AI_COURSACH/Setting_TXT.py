import datetime
import json
import os
import random
import pickle
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Загрузка данных
with open('E:/Courses_Job_AI/AI_COURSACH/DataSets/intents_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

data = data[:5000]
# Собираем patterns и tags (удаляем дубликаты patterns)
patterns = []
tags = []
for item in data:
    unique_patterns = list(set(item['patterns']))  # Удаляем дубли внутри одного тега
    patterns.extend(unique_patterns)
    tags.extend([item['tag']] * len(unique_patterns))

# 2. Токенизация с ограничением словаря
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(patterns)
X = tokenizer.texts_to_sequences(patterns)
X = pad_sequences(X, maxlen=100)  # Фиксированная длина последовательности

# 3. Кодирование меток
le = LabelEncoder()
y = le.fit_transform(tags)
y = to_categorical(y, num_classes=len(le.classes_))

# 4. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 5. Создание модели
model = Sequential([
    Embedding(20000, 64, input_length=100),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Обучение с проверкой памяти
try:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=32,
        epochs=10,
        verbose=1
    )
except Exception as e:
    print(f"Ошибка: {e}")
    print("Пробуем уменьшить batch_size...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=8,  # Уменьшенный batch_size
        epochs=10,
        verbose=1
    )


# 7. Построение графиков
def plot_training_history(history):
    # График точности
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.title('График точности')
    plt.ylabel('Точность')
    plt.xlabel('Эпоха')
    plt.legend()

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.title('График потерь')
    plt.ylabel('Потери')
    plt.xlabel('Эпоха')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Вызов функции для отображения графиков
plot_training_history(history)

# 7. Сохранение модели и токенизатора
randIndex = random.randint(1, 9999)
datefolder = datetime.date.today()
path = f'./Models/Model_{datefolder}_{randIndex}'
os.mkdir(path)
model.save(f'{path}/dialog_model.keras')

with open(f'{path}/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{path}/label_encoder.pickle', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)


# 8. Функция для предсказания
def predict_response(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    proba = model.predict(padded, verbose=0)[0]
    tag = le.inverse_transform([np.argmax(proba)])[0]
    for item in data:
        if item['tag'] == tag:
            return np.random.choice(item['responses'])
    return "Извините, я не понял вопроса."


# Тест
print(predict_response("А ты не боишься?"))