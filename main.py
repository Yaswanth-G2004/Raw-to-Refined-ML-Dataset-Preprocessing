import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 1. Loading of Dataset
# ------------------------------
df = pd.read_csv('Titanic-Dataset.csv')
df = df.copy()  # Avoid chained assignment warnings

# ------------------------------
# 2. Basic Data Overview
# ------------------------------
print("Initial Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# ------------------------------
# 3. Handle Missing Values
# ------------------------------
# Fill missing 'Age' with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing 'Embarked' with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop 'Cabin' due to excessive missing values
df.drop(columns=['Cabin'], inplace=True)

# ------------------------------
# 4. Encode Categorical Variables
# ------------------------------
# Label encode 'Sex'
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# ------------------------------
# 5. Feature Scaling
# ------------------------------
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# ------------------------------
# 6. Outlier Detection and Removal (Fare)
# ------------------------------
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]

# ------------------------------
# 7. Final Dataset Info
# ------------------------------
print("\nFinal Shape:", df.shape)
print("\nSample Data:\n", df.head())

#Saving of cleaned dataset
df.to_csv("titanic_preprocessed_dataset.csv", index=False)



