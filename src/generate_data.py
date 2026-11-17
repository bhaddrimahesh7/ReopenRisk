import os
import numpy as np
import pandas as pd


def create_ticket_dataset(
    n_samples: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    data = {
        "ticket_id": [f"TKT-{10000 + i}" for i in range(n_samples)],
        "category": rng.choice(
            ["Billing", "Technical", "Account Access", "API Integration"],
            size=n_samples,
            p=[0.30, 0.40, 0.20, 0.10],
        ),
        "priority": rng.choice(
            ["Low", "Medium", "High", "Critical"],
            size=n_samples,
            p=[0.25, 0.45, 0.20, 0.10],
        ),
        "first_response_mins": np.round(
            rng.exponential(scale=45, size=n_samples), 1
        ),
        "resolution_hours": np.round(
            rng.gamma(shape=2.0, scale=6.0, size=n_samples), 1
        ),
        "num_messages": rng.integers(2, 25, size=n_samples),
        "escalation_count": rng.choice(
            [0, 1, 2, 3], size=n_samples, p=[0.65, 0.22, 0.10, 0.03]
        ),
        "resolution_note_words": rng.integers(5, 180, size=n_samples),
        "past_customer_tickets": rng.poisson(lam=2.2, size=n_samples),
    }
    df = pd.DataFrame(data)

    # Underlying risk log-odds (imbalanced ~13% positive reopen rate)
    logit = (
        -3.1
        + 0.65 * df["escalation_count"]
        + 0.012 * df["first_response_mins"]
        - 0.015 * df["resolution_note_words"]
        + 0.05 * df["num_messages"]
        + np.where(df["category"] == "API Integration", 0.7, 0.0)
        + np.where(df["priority"] == "Critical", 0.5, 0.0)
        + rng.normal(0, 0.35, size=n_samples)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["reopened"] = (rng.random(n_samples) < prob).astype(int)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/tickets.csv", index=False)
    print(
        f"Saved {len(df)} rows to data/tickets.csv | "
        f"Reopen rate: {df['reopened'].mean():.2%}"
    )
    return df


if __name__ == "__main__":
    create_ticket_dataset()
