# Застосування моделі
import joblib
import pandas as pd

model = joblib.load('model.pkl')

# Завантаження нових даних
new_input = pd.read_csv('new_input.csv')
new_input.drop(columns='Status', inplace=True)

# Побудова передбачень
predictions = model.predict(new_input)

# Збереження передбачень
pd.DataFrame(predictions, columns=['Prediction']).to_csv('predictions.csv', index=False)