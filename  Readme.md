# Оцінка кредитоспроможності клієнтів

## Опис
Цей проєкт має на меті пошук способу оптимізації процесу оцінки кредитоспроможності клієнтів для зниження ризику неповернення кредитів.


## Вимоги
- Python 3.x
- Бібліотеки:
  - pandas
  - numpy
  - scikit-learn
  - joblib

## Структура проєкту
- `variant_1.csv`: Вихідні дані для обробки.
- `new_variant_1.csv`: Оброблені дані після видалення аномалій та заповнення пропущених значень.
- `train_split.csv`: Тренувальний набір даних.
- `model.pkl`: Збережена модель машинного навчання.
- `new_input.csv`: Нові дані для прогнозування.
- `predictions.csv`: Файл з прогнозами, отриманими з навченої моделі.

## Основні функції
1. **Видалення аномалій**: Використовує функцію `find_skewed_boundaries` для виявлення та видалення аномалій з даних.
2. **Заповнення пропущених значень**: Заповнює пропущені значення середнім значенням для числових стовпців та модою для категоріальних.
3. **Кодування категоріальних змінних**: Використовує `LabelEncoder` для перетворення категоріальних змінних у числові.
4. **Стандартизація даних**: Застосовує `StandardScaler` для нормалізації числових змінних.
5. **Розділення даних**: Ділиить дані на тренувальний та тестовий набори.
6. **Навчання моделі**: Використовує логістичну регресію для навчання моделі на тренувальному наборі даних.
7. **Прогнозування**: Застосовує навчену модель для прогнозування на нових даних.

## Використання
1. Завантажте вхідні дані `new_input.csv` у директорію проєкту.
2. Запустіть скрипт `model.py`.
3. Знайдіть результати прогнозування у файлі `predictions.csv`.

