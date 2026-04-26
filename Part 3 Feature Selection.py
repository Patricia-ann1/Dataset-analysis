from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def select_features(df):
    # Prepare X and y
    X = df.drop(columns=['Survived', 'Name', 'Ticket', 'PassengerId'])
    y = df['Survived']
    
    # Use Random Forest to rank importance 
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    return importance.sort_values(by='Importance', ascending=False) [cite: 79]