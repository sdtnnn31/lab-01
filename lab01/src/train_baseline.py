import pandas as pd
from sklearn.metrics import accuracy_score

from common import RESULTS, load_split, make_models, set_seeds


def main():
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "baseline_accuracy.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
