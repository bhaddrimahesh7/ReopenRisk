from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def generate_data(n_samples=2000):
    rng = np.random.default_rng(42)
    category = rng.choice(["Software", "Hardware", "Network", "Access"], size=n_samples)
    agent_experience = rng.integers(1, 6, size=n_samples)
    messages_count = rng.integers(1, 15, size=n_samples)
    resolution_time_hrs = rng.uniform(0.5, 72.0, size=n_samples)

    reopened = (
        (messages_count > 8) | (resolution_time_hrs < 2.0) | (agent_experience == 1)
    ).astype(int)

    df = pd.DataFrame(
        {
            "category": category,
            "agent_experience": agent_experience,
            "messages_count": messages_count,
            "resolution_time_hrs": resolution_time_hrs.round(1),
            "reopened": reopened,
        }
    )
    return df


def main():
    Path("models").mkdir(exist_ok=True)
    df = generate_data()

    X = pd.get_dummies(df.drop(columns=["reopened"]))
    y = df["reopened"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(
        {"model": model, "columns": X_train.columns.tolist()},
        "models/reopenrisk.joblib",
    )
    print("Model trained and saved successfully to models/reopenrisk.joblib!")


if __name__ == "__main__":
    main()
