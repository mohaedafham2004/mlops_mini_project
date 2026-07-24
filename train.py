import mlflow

import mlflow.sklearn

import dagshub

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score

# This line connects my script to my DagsHub-hosted MLflow tracking server

dagshub.init(repo_owner='mohaedafham2004', repo_name='mlops_mini_project', mlflow=True)

# Load a small built-in dataset (flower measurements -> flower species)

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train 3 versions of the model with different settings, so you have real results to compare

for n_estimators, max_depth in [(50, 3), (100, 5), (150, None)]:

    with mlflow.start_run():

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        f1 = f1_score(y_test, preds, average="macro")

        # These log_ calls are what MLflow uses to record the experiment

        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", acc)

        mlflow.log_metric("f1_macro", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"n_estimators={n_estimators}, max_depth={max_depth} -> acc={acc:.3f}, f1={f1:.3f}")
