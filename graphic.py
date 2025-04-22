import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Данные из таблицы пользователя
data = {
    'Threads': [1, 2, 4, 6, 8],
    'Static': [18.23, 8.62, 5.39, 6.45, 4.47],
    'Static, 2': [17.26, 8.64, 5.25, 6.48, 4.48],
    'Dynamic': [17.18, 8.62, 4.83, 4.92, 4.54],
    'Dynamic, 2': [16.25, 8.59, 5.34, 4.53, 4.35],
    'Guided': [16.48, 8.71, 5.77, 5.08, 4.69]
}

df = pd.DataFrame(data)

# Расчет ускорения относительно 1 потока для каждого метода
df_speedup = pd.DataFrame({'Threads': df['Threads']})
for col in df.columns:
    if col != 'Threads':
        # Находим время выполнения для 1 потока для этого метода
        t1 = df[df['Threads'] == 1][col].iloc[0]
        df_speedup[col + '_Speedup'] = t1 / df[col]

# Построение графика "Время выполнения vs. Количество потоков"
plt.figure(figsize=(10, 6))
for col in df.columns:
    if col != 'Threads':
        plt.plot(df['Threads'], df[col], marker='o', linestyle='-', label=col)

plt.xlabel('Количество потоков')
plt.ylabel('Время выполнения (сек)')
plt.title('Зависимость времени выполнения от количества потоков и политики планирования')
plt.xticks(df['Threads'])
plt.grid(True)
plt.legend(title='Политика планирования')
plt.tight_layout()

# Чтобы сохранить или показать график, используйте plt.show() или plt.savefig()
# Например, для сохранения: plt.savefig('execution_time.png')
# Для отображения (если работаете в среде, поддерживающей вывод графиков): plt.show()


# Построение графика "Ускорение vs. Количество потоков"
plt.figure(figsize=(10, 6))
for col in df_speedup.columns:
    if col != 'Threads':
        plt.plot(df_speedup['Threads'], df_speedup[col], marker='o', linestyle='-', label=col.replace('_Speedup', ''))

# Добавляем линию идеального ускорения
plt.plot(df_speedup['Threads'], df_speedup['Threads'], linestyle='--', color='gray', label='Идеальное ускорение')


plt.xlabel('Количество потоков')
plt.ylabel('Ускорение')
plt.title('Зависимость ускорения от количества потоков и политики планирования')
plt.xticks(df_speedup['Threads'])
plt.grid(True)
plt.legend(title='Политика планирования')
plt.tight_layout()

# Чтобы сохранить или показать график, используйте plt.show() или plt.savefig()
# Например, для сохранения: plt.savefig('speedup.png')
plt.show()

# В данном случае, для вывода в интерфейс, используются специальные метки
# Если вы запускаете код локально, закомментируйте эти строки и используйте plt.show() или plt.savefig()
# print(f"@@BEGIN_GRAPH_@@{image_base64_1}@@END_GRAPH_@@") # Это было для внутреннего вывода
# print(f"@@BEGIN_GRAPH_@@{image_base64_2}@@END_GRAPH_@@") # Это было для внутреннего вывода