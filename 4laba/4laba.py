import os
import psycopg2
import pandas as pd
import random
import logging
from cryptography.fernet import Fernet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, text
import seaborn as sns
import matplotlib.pyplot as plt


# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Проверка и генерация ключа для шифрования
if not os.path.exists("secret.key"):
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)
    logging.info("Ключ сгенерирован и сохранен в 'secret.key'")
else:
    logging.info("Файл 'secret.key' уже существует")

# Загрузка ключа из файла
with open("secret.key", "rb") as key_file:
    key = key_file.read()

# Считывание логина и пароля из файла
with open("credentials.txt", "r") as file:
    username = file.readline().strip()
    password = file.readline().strip()

# Шифрование учетных данных
fernet = Fernet(key)
encrypted_username = fernet.encrypt(username.encode())
encrypted_password = fernet.encrypt(password.encode())
print(f'Зашифрованные данные логина: {encrypted_username}')
print(f'Зашифрованные данные пароля: {encrypted_password}')

# Расшифровка учетных данных
decrypted_username = fernet.decrypt(encrypted_username).decode()
decrypted_password = fernet.decrypt(encrypted_password).decode()
print(f'Расшифрованные данные логина: {decrypted_username}')
print(f'Расшифрованные данные пароля: {decrypted_password}')

# Подключение к базе данных PostgreSQL с использованием SQLAlchemy
engine = create_engine(f'postgresql://{username}:{password}@localhost/lab_database')

# Создание таблиц
try:
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS Real_Estate (
                property_id SERIAL PRIMARY KEY,
                type VARCHAR(50),
                area INTEGER,
                price DECIMAL,
                location VARCHAR(100)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS Agents (
                agent_id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                region VARCHAR(100),
                years_experience INTEGER,
                property_id INTEGER REFERENCES Real_Estate (property_id)
            )
        """))
        logging.info("Таблицы созданы в базе данных")
except Exception as e:
    logging.error(f"Ошибка при создании таблиц: {e}")
    raise

# Генерация данных для Real_Estate
real_estates = []
types = ["sale", "rent"]
locations = ["City Center", "Suburbs", "Downtown", "Riverside"]
for i in range(100):
    estate_type = random.choice(types)
    area = random.randint(50, 300)
    price = round(random.uniform(1000, 100000), 2) if estate_type == "sale" else round(random.uniform(500, 5000), 2)
    location = random.choice(locations)
    real_estates.append((estate_type, area, price, location))

# Вставка данных в Real_Estate
try:
    with engine.connect() as conn:
        query = "INSERT INTO Real_Estate (type, area, price, location) VALUES (:type, :area, :price, :location)"
        conn.execute(text(query), [{'type': estate[0], 'area': estate[1], 'price': estate[2], 'location': estate[3]} for estate in real_estates])
        logging.info("Данные вставлены в таблицу Real_Estate")
except Exception as e:
    logging.error(f"Ошибка при вставке данных в Real_Estate: {e}")
    raise

# Генерация данных для Agents
agents = []
regions = ["North", "South", "East", "West"]
property_ids = range(1, 101)  # Генерация списка существующих property_id
for i in range(50):
    name = f"Agent_{i}"
    region = random.choice(regions)
    years_experience = random.randint(1, 20)
    property_id = random.choice(property_ids)  # Теперь случайный выбор property_id из существующих
    agents.append((name, region, years_experience, property_id))

# Вставка данных в Agents
try:
    with engine.connect() as conn:
        query = "INSERT INTO Agents (name, region, years_experience, property_id) VALUES (:name, :region, :years_experience, :property_id)"
        conn.execute(text(query), [{'name': agent[0], 'region': agent[1], 'years_experience': agent[2], 'property_id': agent[3]} for agent in agents])
        logging.info("Данные вставлены в таблицу Agents")
except Exception as e:
    logging.error(f"Ошибка при вставке данных в Agents: {e}")
    raise

# Извлечение данных из Real_Estate
try:
    query = "SELECT type, area, price, location FROM Real_Estate"
    df = pd.read_sql_query(query, engine)
    logging.info("Данные успешно извлечены из базы данных для анализа")

    # Расчет средней цены за квадратный метр по типам недвижимости
    df['price_per_sqm'] = df['price'] / df['area']
    average_price_per_type = df.groupby('type')['price_per_sqm'].mean()
    print("Средняя цена за квадратный метр по типам недвижимости:")
    print(average_price_per_type)

    # Извлечение данных об агентах
    query_agents = "SELECT * FROM Agents"
    df_agents = pd.read_sql_query(query_agents, engine)
    logging.info("Данные об агентах успешно извлечены из базы данных")

except Exception as e:
    logging.error(f"Ошибка при извлечении данных: {e}")
    raise



# Обработка данных и подготовка к обучению модели
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
df['location'] = le.fit_transform(df['location'])

X = df[['area', 'price', 'location']]
y = df['type']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели K-ближайших соседей
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Предсказания и оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')  # Убедитесь, что используется правильный параметр average
print(f'Accuracy: {accuracy:.2f}')
print(f'F1 Score: {f1:.2f}')

# Убедимся, что 'y_test' и 'y_pred' являются числовыми
y_test = pd.Series(y_test).astype(float)
y_pred = pd.Series(y_pred).astype(float)

# Преобразование типа недвижимости в числовой тип для boxplot
df['price'] = df['price'].astype(float)
df['area'] = df['area'].astype(float)

# Определение количества строк для отображения (20% от общего числа)
num_samples = int(len(X_test) * 0.2)

# Отбор первых 20% данных
num_samples = int(len(X_test) * 0.2)  # Определение количества строк для отображения (20% от общего числа)
X_test_subset = X_test.iloc[:num_samples]
y_pred_subset = y_pred[:num_samples]

# Построение графика
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_test_subset['area'], X_test_subset['price'], c=y_pred_subset, cmap='coolwarm', alpha=0.7, edgecolors='k')

# Добавление красной диагональной линии
plt.plot([0, X_test_subset['area'].max()],
         [0, 7000],  # Здесь 7000 - это максимальное значение цены, которое вы установили ранее
         color='red', linestyle='--', linewidth=2, label='Идеальные значения')

plt.colorbar(scatter, ticks=[0, 1], label='Тип (0: Аренда, 1: Продажа)')
plt.xlabel('Площадь (кв. м.)')
plt.ylabel('Цена (руб.)')
plt.title('Предсказанные типы недвижимости по площади и цене (первые 20%)')

# Установка пределов оси Y до 15000
plt.ylim(0, 7000)  # Изменяем максимальное значение цены на 7000
plt.xlim(0, X_test_subset['area'].max())  # Устанавливаем пределы по оси X

plt.grid()
plt.legend()  # Добавление легенды для линии
plt.show()
