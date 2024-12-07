import praw
import pandas as pd
import re
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import dash
from dash import dcc, html
from dash.dash_table import DataTable
import plotly.express as px
import plotly.graph_objs as go

# Загрузка данных для предобработки текста
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Настройка API Reddit
with open("6laba_client_id_secret.txt", "r") as file:
    client_id = file.readline().strip()  # Чтение первого значения (client_id)
    client_secret = file.readline().strip()  # Чтение второго значения (client_secret)
    user_agent = file.readline().strip()  # Чтение третьего значения (user_agent)

# Инициализация Reddit API
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Сбор данных с Reddit
def fetch_reddit_data(subreddit_name, query, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.search(query, limit=limit):
        posts.append({
            "title": submission.title,
            "selftext": submission.selftext,
            "created_utc": submission.created_utc,
            "score": submission.score
        })
    return pd.DataFrame(posts) #СБОР ДАННЫХ

# Предобработка текста
def preprocess_text_basic(text):
    text = re.sub(r'http\S+', '', text)  # Удаление ссылок
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Удаление спецсимволов
    text = text.lower()  # Приведение к нижнему регистру
    tokens = text.split()  # Разделение по пробелам
    tokens = [word for word in tokens if word not in stop_words]  # Удаление стоп-слов
    return ' '.join(tokens) #УДАЛЯЕТ СТОП СЛОВ

# Анализ тональности
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if not text:
        return 0  # Если текст пустой
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound'] #ОЦЕНКА ТОНАЛЬНОСТИ

# Основная программа
def main():
    # Шаг 1: Сбор данных
    print("Сбор данных с Reddit...")
    data = fetch_reddit_data("politics", "elections", limit=200)
    print(f"Собрано {len(data)} постов.")

    # Шаг 2: Предобработка текста
    print("Предобработка текста...")
    data['cleaned_text'] = data['selftext'].apply(preprocess_text_basic)

    # Шаг 3: Анализ тональности
    print("Анализ тональности...")
    data['sentiment'] = data['cleaned_text'].apply(analyze_sentiment)

    # Преобразуем 'created_utc' в формат datetime и создаем колонку 'date'
    data['created_utc'] = pd.to_datetime(data['created_utc'], unit='s')
    data['date'] = data['created_utc'].dt.date  # Создаем колонку 'date' для группировки

    # Шаг 4: Тематическое моделирование
    print("Построение LDA-модели...")
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    data_matrix = vectorizer.fit_transform(data['cleaned_text'])
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(data_matrix)

    # Вывод ключевых слов по темам
    print("\nКлючевые слова по темам:")
    for idx, topic in enumerate(lda_model.components_):
        print(f"Тема {idx+1}:")
        print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

    # Шаг 5: Визуализация
    print("Визуализация данных...")
    daily_sentiment = data.groupby(data['date'])['sentiment'].mean().reset_index()

    # Визуализация распределения тональности
    fig1 = px.histogram(data, x="sentiment", nbins=30, title="Распределение тональности")
    fig1.update_layout(
        xaxis_title="Тональность",
        yaxis_title="Частота"
    )

    # Визуализация изменения настроений с течением времени
    fig2 = go.Figure(data=[go.Scatter(
        x=daily_sentiment['date'],
        y=daily_sentiment['sentiment'],
        mode='lines',
        name='Средняя тональность'
    )])
    fig2.update_layout(
        title="Изменение настроений с течением времени",
        xaxis_title="Дата",
        yaxis_title="Средняя тональность"
    )

    # Создаем дашборд с Dash
    app = dash.Dash(__name__)

    # Ограничиваем количество символов в selftext
    data['title'] = data['title'].apply(lambda x: x[:150] + '...' if len(x) > 50 else x)
    data['selftext'] = data['selftext'].apply(lambda x: x[:50] + '...' if len(x) > 15 else x)

    # Добавляем нумерацию постов
    data['Post #'] = range(1, len(data) + 1)

    # Создание таблицы
    app.layout = html.Div([
        html.H1("Анализ общественного мнения на Reddit"),
        html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2)
        ]),
        html.Div([
            DataTable(
                id='table',
                columns=[
                    {"name": "Post #", "id": "Post #"},
                    {"name": "Title", "id": "title"},
                    {"name": "Selftext", "id": "selftext"},
                    {"name": "Date", "id": "created_utc"}
                ],
                data=data[['Post #', 'title', 'selftext', 'created_utc']].to_dict('records'),
                style_table={'height': '400px', 'overflowY': 'auto'},  # Прокрутка таблицы
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},  # Тема для заголовков
                style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},  # Цвет фона данных
            )
        ])
    ])

    app.run_server(debug=True)

if __name__ == "__main__":
    main()
