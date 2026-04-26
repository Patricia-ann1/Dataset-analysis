import pandas as pd
import numpy as np

def engineer_features(df):
    # 1. Derived Features [cite: 47]
    # Family Size and IsAlone [cite: 51, 52]
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Title Extraction from Name [cite: 53]
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Fare per Person [cite: 56]
    df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']
    
    # Age Groups [cite: 55]
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 19, 59, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])
    
    # 2. Categorical Encoding [cite: 57]
    # One-hot encode nominal features [cite: 58]
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'AgeGroup'], drop_first=True)
    
    # 3. Transformations [cite: 62]
    # Log transform skewed Fare [cite: 64]
    df['Fare'] = np.log1p(df['Fare'])
    
    return df