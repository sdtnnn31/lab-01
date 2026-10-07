import statistics
import time

import joblib
import pandas as pd
import psutil
from memory_profiler import memory_usage
from sklearn.base import clone

from common import RESULTS, load_split, make_models, set_seeds

TRAIN_RUNS = 5
INFER_RUNS = 100


def median_train_time(model, X, y):
    clone(model).fit(X, y)
    times = []
    for _ in range(TRAIN_RUNS):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X, y)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def median_latency(model, x):
    model.predict(x)
    times = []
    for _ in range(INFER_RUNS):
        t0 = time.perf_counter()
        model.predict(x)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def rss_mb():
    return psutil.Process().memory_info().rss / 2**20


def peak_mb(func):
    base = rss_mb()
    peak = memory_usage((func, (), {}), interval=0.001, max_usage=True)
    if isinstance(peak, (list, tuple)):
        peak = max(peak)
    return peak, max(0.0, peak - base)


def main():
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    sample = X_test[:1]
    rows = []
    for name, model in make_models().items():
        train_s = median_train_time(model, X_train, y_train)
        fitted = clone(model).fit(X_train, y_train)
        latency_s = median_latency(fitted, sample)

        path = RESULTS / f"model_{name}.joblib"
        joblib.dump(fitted, path)
        size_bytes = path.stat().st_size

        train_peak, train_delta = peak_mb(lambda: clone(model).fit(X_train, y_train))

        def infer():
            for _ in range(INFER_RUNS):
                fitted.predict(sample)

        infer_peak, infer_delta = peak_mb(infer)

        rows.append({
            "model": name,
            "train_time_ms_median": round(train_s * 1000, 3),
            "latency_ms_median": round(latency_s * 1000, 4),
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
            "train_peak_rss_mb": round(train_peak, 2),
            "train_delta_mb": round(train_delta, 2),
            "infer_peak_rss_mb": round(infer_peak, 2),
            "infer_delta_mb": round(infer_delta, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "measurements.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
