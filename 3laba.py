import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px

df = None
data_loaded = False

try:
    df = pd.read_csv('arenda_nedv.csv')

    df.columns = df.columns.str.strip()
    print("Столбцы CSV файла:", df.columns)

    data_loaded = True

except FileNotFoundError:
    print("Файл не найден. Убедитесь, что путь к файлу правильный.")
except pd.errors.EmptyDataError:
    print("Файл пуст. Проверьте содержимое файла.")
except Exception as e:
    print(f"Произошла ошибка при загрузке данных: {e}")

app = dash.Dash(__name__)

if data_loaded:
    app.layout = html.Div([
        html.H1("Анализ данных о недвижимости"),

        # Фильтры
        html.Label('Выберите тип недвижимости:'),
        dcc.Dropdown(
            id='property-type-dropdown',
            options=[{'label': prop_type, 'value': prop_type} for prop_type in df['Property_Type'].unique()],
            value=None,
            placeholder="Выберите тип недвижимости"
        ),

        html.Label('Выберите диапазон цен (сом):'),
        dcc.RangeSlider(
            id='price-range-slider',
            min=df['Price_som'].min(),
            max=df['Price_som'].max(),
            step=10000,
            value=[df['Price_som'].min(), df['Price_som'].max()],
            marks={int(price): str(price) for price in range(0, int(df['Price_som'].max()), 5000000)}
        ),

        html.Label('Минимальная площадь (м²):'),
        dcc.Input(id='min-area-input', type='number', value=0, placeholder="Введите минимальную площадь"),

        # Графики
        dcc.Graph(id='price-histogram'),
        dcc.Graph(id='area-histogram'),
        dcc.Graph(id='property-type-pie'),
        dcc.Graph(id='price-area-scatter')
    ])
else:
    app.layout = html.Div([
        html.H1("Ошибка загрузки данных"),
        html.P("Не удалось загрузить данные о недвижимости. Проверьте файл CSV.")
    ])

@app.callback(
    [Output('price-histogram', 'figure'),
     Output('area-histogram', 'figure'),
     Output('property-type-pie', 'figure'),
     Output('price-area-scatter', 'figure')],
    [Input('property-type-dropdown', 'value'),
     Input('price-range-slider', 'value'),
     Input('min-area-input', 'value')]
)
def update_graphs(selected_property, price_range, min_area):
    if not data_loaded:
        return {}, {}, {}, {}

    try:
        # Фильтрация данных
        filtered_df = df.copy()

        if selected_property:
            filtered_df = filtered_df[filtered_df['Property_Type'] == selected_property]

        if price_range:
            filtered_df = filtered_df[(filtered_df['Price_som'] >= price_range[0]) & (filtered_df['Price_som'] <= price_range[1])]

        if min_area is not None:
            filtered_df = filtered_df[filtered_df['Area_m'] >= min_area]

        if filtered_df.empty:
            raise ValueError("Нет данных для выбранных фильтров.")

        # Построение графиков на основе фильтрованных данных
        price_histogram = px.histogram(filtered_df, x='Price_som', nbins=10, title='Распределение цен недвижимости')

        area_histogram = px.histogram(filtered_df, x='Area_m', nbins=10, title='Распределение площадей объектов')

        property_type_pie = px.pie(filtered_df, names='Property_Type', title='Распределение по типам недвижимости')

        price_area_scatter = px.scatter(filtered_df, x='Area_m', y='Price_som',
                                        title='Зависимость цены от площади',
                                        labels={'Area_m': 'Площадь (м²)', 'Price_som': 'Цена (сом)'})

        return price_histogram, area_histogram, property_type_pie, price_area_scatter

    except ValueError as e:
        return px.histogram(), px.histogram(), px.pie(), px.scatter(), f"Ошибка: {str(e)}"
    except Exception as e:
        return px.histogram(), px.histogram(), px.pie(), px.scatter(), f"Произошла ошибка: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True)
