import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import load_data, split_data, normalize_data

def main():
    # Charger les données
    print("Chargement des données...")
    df = load_data('data/sample.csv')
    print(df)
    
    # Séparer X et y
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Diviser train/test
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Normalisation (de DS B)
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
    
    # Random Forest avec hyperparamètres (de DS A)
    print("Entraînement du modèle...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Évaluer
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy finale: {accuracy:.2f}")

if __name__ == "__main__":
    main()