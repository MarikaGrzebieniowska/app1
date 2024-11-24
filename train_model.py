import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

data = pd.read_csv('DSP_1.csv')

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(2, inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[features]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

print("Training GridSearchCV...")
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Best model accuracy: {accuracy_score(y_test, y_pred):.2f}")

with open('model.sv', 'wb') as f:
    pickle.dump(best_model, f)

print("DONE: model trained and saved to 'model.sv'.")
