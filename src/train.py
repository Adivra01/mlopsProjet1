import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import load_data, split_data, normalize_data

def main():
    # Configuration de l'expérience MLflow
    mlflow.set_experiment("Mini-Projet-MLOps")
    
    with mlflow.start_run():
        # Charger les données
        print("Chargement des données...")
        df = load_data('data/sample.csv')
        print(df)
        
        # Séparer X et y
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Diviser train/test
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Normalisation
        X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
        
        # Modèle avec plus d'arbres (200 au lieu de 100)
        print("Entraînement du modèle...")
        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Évaluer
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        
        # Log dans MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()