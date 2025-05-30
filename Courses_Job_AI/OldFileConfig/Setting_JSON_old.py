import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

def extract_qa_pairs(json_data):
    qa_pairs = []
    for dialog in json_data:
        dialog_turns = dialog["dialog"]
        for i in range(len(dialog_turns) - 1):
            if dialog_turns[i]["speaker"] == "user" and dialog_turns[i+1]["speaker"] == "assistant":
                qa_pairs.append({
                    "question": dialog_turns[i]["text"],
                    "answer": dialog_turns[i+1]["text"]
                })
    return qa_pairs


# Загрузка JSON (может быть список файлов)
with open('./DataSets/dialog_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

qa_pairs = extract_qa_pairs(data)
print(f"Извлечено {len(qa_pairs)} пар вопрос-ответ.")

train_qa, test_qa = train_test_split(qa_pairs, test_size=0.2, random_state=42)

questions = [pair["question"] for pair in qa_pairs]
answers = [pair["answer"] for pair in qa_pairs]

# Векторизация текста
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Поиск ближайшего соседа
model = NearestNeighbors(n_neighbors=1, metric="cosine")
model.fit(X)

# Функция для предсказания ответа
def get_response(user_query):
    query_vec = vectorizer.transform([user_query])
    distances, indices = model.kneighbors(query_vec)
    return answers[indices[0][0]]


# Тестирование
user_query = "Я хочу посетить остров с хорошим пляжем. Вы знаете какой-нибудь?"
response = get_response(user_query)
print(response)  # Выведет соответствующий ответ

'''------------------------------------------------------------------'''
'''----------------Графики обучения модели---------------------------------'''
'''------------------------------------------------------------------'''
# 1. Визуализация распределения вопросов в пространстве признаков (после уменьшения размерности)
def plot_questions_embeddings(X, sample_size=500):
    # Уменьшаем размерность для визуализации
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())

    # Берем подвыборку для визуализации
    if len(X_reduced) > sample_size:
        indices = np.random.choice(len(X_reduced), sample_size, replace=False)
        X_reduced = X_reduced[indices]

    plt.figure(figsize=(10, 8))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
    plt.title('Визуализация вопросов в 2D пространстве (PCA)')
    plt.xlabel('Компонента 1')
    plt.ylabel('Компонента 2')
    plt.show()


# 2. Гистограмма расстояний для примеров из тестового набора
def plot_distance_distribution(test_questions, model, vectorizer, sample_size=100):
    if len(test_questions) > sample_size:
        test_questions = np.random.choice(test_questions, sample_size, replace=False)

    distances = []
    for question in test_questions:
        query_vec = vectorizer.transform([question])
        dist, _ = model.kneighbors(query_vec)
        distances.append(dist[0][0])

    plt.figure(figsize=(10, 6))
    sns.histplot(distances, bins=20, kde=True)
    plt.title('Распределение расстояний до ближайшего соседа')
    plt.xlabel('Косинусное расстояние')
    plt.ylabel('Частота')
    plt.show()


# 3. Примеры работы модели с визуализацией схожести
def plot_example_comparisons(test_qa, model, vectorizer, n_examples=3):
    examples = np.random.choice(test_qa, n_examples, replace=False)

    for example in examples:
        query = example["question"]
        query_vec = vectorizer.transform([query])
        distances, indices = model.kneighbors(query_vec)

        closest_question = questions[indices[0][0]]
        similarity = 1 - distances[0][0]  # преобразуем расстояние в схожесть

        plt.figure(figsize=(10, 4))
        plt.bar(['Схожесть'], [similarity], color='skyblue')
        plt.ylim(0, 1)
        plt.title(
            f'Схожесть запроса с ближайшим вопросом\nЗапрос: "{query[:50]}..."\nБлижайший вопрос: "{closest_question[:50]}..."')
        plt.ylabel('Косинусная схожесть')
        plt.show()


# Вызов функций визуализации
plot_questions_embeddings(X[:5000])
plot_distance_distribution([pair["question"] for pair in test_qa], model, vectorizer)
plot_example_comparisons(test_qa, model, vectorizer)


def evaluate_model(test_qa, model, vectorizer):
    correct = 0
    similarities = []

    for pair in test_qa:
        query_vec = vectorizer.transform([pair["question"]])
        distances, indices = model.kneighbors(query_vec)

        # Проверяем, совпадает ли ответ с ожидаемым (это очень строгая метрика)
        if answers[indices[0][0]] == pair["answer"]:
            correct += 1
        similarities.append(1 - distances[0][0])

    accuracy = correct / len(test_qa)
    avg_similarity = np.mean(similarities)

    print(f"Точность (строгая): {accuracy:.2%}")
    print(f"Средняя схожесть с ближайшим вопросом: {avg_similarity:.2f}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(['Точность'], [accuracy], color='lightgreen')
    plt.ylim(0, 1)
    plt.title('Точность модели')

    plt.subplot(1, 2, 2)
    sns.histplot(similarities, bins=20, kde=True)
    plt.title('Распределение схожестей')
    plt.tight_layout()
    plt.show()


evaluate_model(test_qa, model, vectorizer)