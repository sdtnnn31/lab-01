import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS = LAB_DIR / "results"
RESULTS.mkdir(exist_ok=True)


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)


def make_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }
