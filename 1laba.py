import pandas as pd
df = pd.read_csv('arenda_nedv.csv')
df['Price_per_m'] = df['Price_som'] / df['Area_m']

average_price_per_m = df.groupby('Property_Type')['Price_per_m'].mean()
total_price_by_type = df.groupby('Property_Type')['Price_som'].sum()

average_price_per_m = average_price_per_m.map("{:.3f}".format)

print("Средняя цена за квадратный метр по типам недвижимости:")
for property_type, price in average_price_per_m.items():
    print(f"{property_type}: {price}")

print("\nОбщая стоимость объектов по типам недвижимости:")
for property_type, total_price in total_price_by_type.items():
    print(f"{property_type}: {total_price}")

