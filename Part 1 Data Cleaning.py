import pandas as pd
import numpy as np

def clean_titanic_data(df):
    # 1. Missing Value Handling [cite: 29, 32]
    # Age: Impute with median to avoid outlier influence [cite: 33, 37]
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Embarked: Impute with most frequent value (mode) [cite: 33]
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Cabin: Too many missing values, drop the column [cite: 35]
    df.drop(columns=['Cabin'], inplace=True)
    
    # 2. Data Consistency [cite: 39]
    # Ensure 'Sex' is uniform (male/female) [cite: 40]
    df['Sex'] = df['Sex'].str.lower().str.strip()
    
    # 3. Outlier Handling [cite: 36]
    # Cap 'Fare' at the 99th percentile to handle extreme outliers [cite: 37, 38]
    upper_limit = df['Fare'].quantile(0.99)
    df['Fare'] = np.where(df['Fare'] > upper_limit, upper_limit, df['Fare'])
    
    # 4. Remove duplicates [cite: 41, 42]
    df = df.drop_duplicates()
    
    return df

# Load and process
train_df = pd.read_csv('data/train.csv') [cite: 12]
train_cleaned = clean_titanic_data(train_df)
train_cleaned.to_csv('data/train_cleaned.csv', index=False) [cite: 44]