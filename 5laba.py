import pandas as pd
import numpy as np
import random
from faker import Faker
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import schedule
import time

# Настройки подключения к базе данных
with open("5credentials.txt", "r") as file:
    DB_NAME = file.readline().strip()
    USER = file.readline().strip()
    PASSWORD = file.readline().strip()
    HOST = file.readline().strip()
    PORT = file.readline().strip()

# Создаем подключение к базе данных через SQLAlchemy
engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}')
fake = Faker()

# 1. Создание таблиц и функции
def setup_database():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS "Real_Estate" (
                property_id SERIAL PRIMARY KEY,
                type VARCHAR(50),
                area INTEGER,
                price DECIMAL,
                location VARCHAR(100)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS "Agents" (
                agent_id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                region VARCHAR(100),
                years_experience INTEGER,
                property_id INTEGER,
                CONSTRAINT fk_property FOREIGN KEY (property_id)
                REFERENCES "Real_Estate" (property_id)
            )
        """))
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION dynamic_query_executor(query_condition TEXT)
            RETURNS TABLE(property_id INT, type VARCHAR, area INT, price DECIMAL, location VARCHAR) AS $$
            BEGIN
                RETURN QUERY EXECUTE format('SELECT * FROM "Real_Estate" WHERE %s', query_condition);
            END;
            $$ LANGUAGE plpgsql;
        """))
        print("База данных настроена.")


# 2. Массовая вставка данных
def batch_insert_data(batch_size=1000):
    data_real_estate = []
    data_agents = []

    with engine.begin() as conn:  # Транзакция для массовой вставки
        try:
            for _ in range(10000):  # Генерируем 10,000 записей
                # Генерация данных для таблицы Real_Estate
                property_type = random.choice(['House', 'Apartment', 'Condo'])
                area = random.randint(50, 300)
                price = round(random.uniform(10000, 500000), 2)
                location = fake.address()

                # Добавляем данные в буфер
                data_real_estate.append({
                    'type': property_type,
                    'area': area,
                    'price': price,
                    'location': location
                })

                if len(data_real_estate) >= batch_size:
                    # Преобразование в DataFrame и вставка данных в Real_Estate
                    df_real_estate = pd.DataFrame(data_real_estate)
                    df_real_estate.to_sql('Real_Estate', conn, if_exists='append', index=False)
                    data_real_estate = []  # Очищаем буфер

                    result = conn.execute(
                        text('SELECT property_id FROM "Real_Estate" ORDER BY property_id DESC LIMIT :limit'),
                        {'limit': batch_size}
                    ).mappings()  # Преобразует строки в словари

                    property_ids = [row['property_id'] for row in result]  # Доступ по ключу 'property_id'

                    # Создаем данные для таблицы Agents
                    for property_id in property_ids:
                        agent_name = fake.name()
                        agent_region = fake.city()
                        years_experience = random.randint(1, 30)
                        data_agents.append({
                            'name': agent_name,
                            'region': agent_region,
                            'years_experience': years_experience,
                            'property_id': property_id
                        })

                    # Вставка данных в таблицу Agents
                    if data_agents:
                        df_agents = pd.DataFrame(data_agents)
                        df_agents.to_sql('Agents', conn, if_exists='append', index=False)
                        data_agents = []  # Очищаем буфер

            # Оставшиеся данные в буфере вставляем в таблицы
            if data_real_estate:
                df_real_estate = pd.DataFrame(data_real_estate)
                df_real_estate.to_sql('Real_Estate', conn, if_exists='append', index=False)

            if data_agents:
                df_agents = pd.DataFrame(data_agents)
                df_agents.to_sql('Agents', conn, if_exists='append', index=False)

            print("Массовая загрузка данных завершена.")
        except Exception as e:
            print(f"Ошибка массовой загрузки данных: {e}")



# 3. Загрузка данных с использованием динамического SQL
def load_data_with_execute(condition="price > 50000"):
    query = f'SELECT * FROM "Real_Estate" WHERE {condition}'  # Убедитесь, что кавычки соответствуют имени таблицы
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

# 4. Разделение данных
def split_data(df):
    X = df[['area', 'price']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 5. Обучение модели регрессии
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Среднеквадратичная ошибка: {mse}")
    return y_test, y_pred

# 6. Обучение классификационной модели
def train_classification_model(df):
    # Шаг 1: Кодирование категориальных переменных
    le = LabelEncoder()
    # Кодируем столбец 'type' (тип недвижимости) в числовые метки
    df['type_encoded'] = le.fit_transform(df['type'])

    # Шаг 2: Разделяем данные на признаки и целевую переменную
    X = df[['area', 'price']]  # Признаки: площадь и цена
    y = df['type_encoded']     # Целевая переменная: закодированный тип недвижимости

    # Шаг 3: Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Шаг 4: Обучение модели
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Шаг 5: Предсказание на тестовых данных
    y_pred = model.predict(X_test)

    # Шаг 6: Оценка точности
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Шаг 7: Вывод отчета по классификации
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, le

# 7. Визуализация данных
def visualize_results(y_test, y_pred, noise_level=0.02):
    # Преобразуем y_test и y_pred в numpy массивы, если это не так
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Определяем размер выборки (1% данных)
    sample_size = max(1, int(len(y_test) * 0.01))  # Минимум 1 точка
    indices = np.random.choice(len(y_test), size=sample_size, replace=False)

    # Выбираем подмножество данных
    y_test_sample = y_test[indices]
    y_pred_sample = y_pred[indices]

    # Добавляем шум к точкам для разброса (только для визуализации)
    y_test_sample_noisy = y_test_sample + np.random.uniform(-noise_level, noise_level, size=sample_size)
    y_pred_sample_noisy = y_pred_sample + np.random.uniform(-noise_level, noise_level, size=sample_size)

    # Визуализация
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test_sample_noisy, y=y_pred_sample_noisy, color='blue', label='Предсказанные данные', alpha=0.7)
    sns.scatterplot(x=y_test_sample, y=y_test_sample, color='red', label='Фактические данные', marker='.', alpha=0.7)

    plt.xlabel("Фактическая цена", fontsize=12)
    plt.ylabel("Предсказанная цена", fontsize=12)
    plt.title("Фактическая vs Предсказанная цена", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()  # Автоматически подгоняет элементы графика
    plt.show()

def visualize_distribution(df):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df['price'], bins=50, kde=True, color='blue')
    plt.title("Распределение цен")

    plt.subplot(1, 2, 2)
    sns.histplot(df['area'], bins=50, kde=True, color='green')
    plt.title("Распределение площади")

    plt.tight_layout()
    plt.show()

# 8. Автоматическое обновление данных
def update_data():
    with engine.begin() as conn:  # Используем begin для управления транзакцией
        try:
            # Очищаем таблицы перед обновлением данных
            conn.execute(text('TRUNCATE TABLE "Agents" RESTART IDENTITY CASCADE'))
            conn.execute(text('TRUNCATE TABLE "Real_Estate" RESTART IDENTITY CASCADE'))

            new_real_estate_data = []  # Для хранения данных недвижимости
            new_agent_data = []        # Для хранения данных агентов

            for _ in range(10000):  # Генерация 10,000 записей
                # Генерация данных для Real_Estate
                property_type = random.choice(['House', 'Apartment', 'Condo', 'Villa'])
                area = random.randint(30, 500)
                price = round(random.uniform(10000.0, 1000000.0), 2)
                location = fake.city()

                # Добавление данных в список
                new_real_estate_data.append({
                    'type': property_type,
                    'area': area,
                    'price': price,
                    'location': location
                })

            # Массовая вставка данных в Real_Estate
            conn.execute(
                text("""
                    INSERT INTO "Real_Estate" (type, area, price, location)
                    VALUES (:type, :area, :price, :location)
                """),
                new_real_estate_data
            )

            # Получаем все сгенерированные property_id
            real_estate_ids = conn.execute(
                text('SELECT property_id FROM "Real_Estate"')
            ).fetchall()

            for property_id in real_estate_ids:
                # Генерация данных для Agents
                agent_name = fake.name()
                agent_region = fake.city()
                years_experience = random.randint(1, 30)

                # Добавление данных в список
                new_agent_data.append({
                    'name': agent_name,
                    'region': agent_region,
                    'years_experience': years_experience,
                    'property_id': property_id.property_id  # Используем полученный ID
                })

            # Массовая вставка данных в Agents
            conn.execute(
                text("""
                    INSERT INTO "Agents" (name, region, years_experience, property_id)
                    VALUES (:name, :region, :years_experience, :property_id)
                """),
                new_agent_data
            )

            print("Таблицы успешно очищены и обновлены 10,000 новыми записями.")
        except Exception as e:
            print(f"Ошибка обновления данных: {e}")




# Основной блок
if __name__ == "__main__":
    setup_database()
    batch_insert_data()

    df = load_data_with_execute("price > 50000")
    X_train, X_test, y_train, y_test = split_data(df)
    y_test, y_pred = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    train_classification_model(df)
    visualize_results(y_test, y_pred)
    visualize_distribution(df)

    schedule.every(10).seconds.do(update_data)
    while True:
        schedule.run_pending()
        time.sleep(1)
