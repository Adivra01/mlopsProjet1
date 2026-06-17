import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import load_data, split_data

df = load_data('data/sample.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = split_data(X, y)

# Random Forest avec hyperparamètres
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy Random Forest: {accuracy_score(y_test, y_pred):.2f}")