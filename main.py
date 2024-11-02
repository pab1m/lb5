import joblib
import os.path
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from processing_anomalies import find_skewed_boundaries
from model_selection import model_training


warnings.simplefilter('ignore')
print(os.path.exists("variant_1.csv"))
df = pd.read_csv("variant_1.csv")
df.drop(columns=[df.columns[0], 'year'], inplace=True)


def remove_anomalies(df, columns):
    for column in columns:
        lower_bound, upper_bound = find_skewed_boundaries(df, column)
        anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        df = df.drop(anomalies.index)
    return df

columns_to_check = ['income', 'loan_amount', 'Upfront_charges', 'property_value', 'LTV', 'rate_of_interest']
df = remove_anomalies(df, columns_to_check)


# Список стовпців, для яких потрібно заповнити пропущені значення середнім
columns_to_fill_mean = [
    'rate_of_interest',
    'Interest_rate_spread',
    'Upfront_charges',
    'property_value',
    'income',
    'LTV'
]

for column in columns_to_fill_mean:
    df[column] = df[column].fillna(df[column].mean())

df['term'] = df['term'].fillna(df['term'].max())
df['loan_limit'] = df['loan_limit'].fillna(df['loan_limit'].mode()[0])

# Видалення рядків з пропущеними значеннями у вказаних колонках
df = df.dropna(subset=['approv_in_adv', 'loan_purpose', 'Neg_ammortization', 'age', 'submission_of_application'])

df.replace({
    'age': {
        '<25': 1,
        '25-34': 2,
        '35-44': 2,
        '45-54': 4,
        '55-64': 5,
        '65-74': 6,
        '>74': 7,
    },

}, inplace=True)


label_encoder = LabelEncoder()
# Список стовпців, для яких потрібно застосувати LabelEncoder
columns_to_encode = [
    'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
    'Credit_Worthiness', 'open_credit', 'business_or_commercial', 'Neg_ammortization',
    'interest_only', 'lump_sum_payment', 'construction_type', 'occupancy_type',
    'Secured_by', 'total_units', 'credit_type', 'co-applicant_credit_type',
    'submission_of_application', 'Region', 'Security_Type'
]

# Застосування LabelEncoder до кожного стовпця зі списку
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])


scaler = StandardScaler()
columns_to_standardize = ['loan_amount', 'Upfront_charges', 'property_value', 'income', 'term', 'Credit_Score', 'LTV']
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

df.to_csv('new_variant_1.csv', index=False)


# Завантаження даних
data = pd.read_csv('new_variant_1.csv')

# Розділення даних
train, new_input = train_test_split(data, test_size=0.1, random_state=42)

# Збереження розділених даних
train.to_csv('train_split.csv', index=False)
new_input.to_csv('new_input.csv', index=False)


# Навчання та оцінка моделі
df = pd.read_csv('train_split.csv')
model_training(df)

