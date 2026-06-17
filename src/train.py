import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import load_data, split_data, normalize_data

df = load_data('data/sample.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = split_data(X, y)

X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"Accuracy Logistic Regression: {accuracy_score(y_test, y_pred):.2f}")