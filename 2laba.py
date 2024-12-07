import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('arenda_nedv.csv')

print(f"Средняя цена: {df['Price_som'].mean():.2f} сом.")
print(f"Средняя площадь: {df['Area_m'].mean():.2f} м²")
print(f"Медиана цены: {df['Price_som'].median()} сом.")
print(f"Мода типов объектов: {df['Property_Type'].mode()[0]}")
print(f"Стандартное отклонение площади: {df['Area_m'].std():.2f} м²")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
df['Price_som'].plot(kind='hist', bins=10, color='blue', alpha=0.7, title='Распределение цен')
plt.xlabel('Цена (сом)')

plt.subplot(1, 2, 2)
df['Area_m'].plot(kind='hist', bins=10, color='green', alpha=0.7, title='Распределение площадей')
plt.xlabel('Площадь (м²)')

plt.tight_layout()
plt.show()

df['Property_Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6), title='Распределение объектов по типам')
plt.ylabel('')
plt.show()