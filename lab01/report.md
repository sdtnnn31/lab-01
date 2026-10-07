# Lab 01: Environment and first system measurements

## 1. Goal

## 2. Method

- Environment: Python 3.11.8, pinned `requirements.txt`, versions in `results/versions.txt`
- Data: Breast Cancer Wisconsin, 70/30 stratified split, `random_state=42`
- Models: LogisticRegression(max_iter=1000), RandomForestClassifier(n_estimators=100)
- Training time: 1 warm-up + median of 5 runs
- Inference: single sample, 1 warm-up + median of 100 runs
- Model size: `joblib.dump`
- Peak memory: `memory_profiler` (process RSS peak and delta above baseline)

## 3. Results

### Accuracy

| Model | Test accuracy |
|---|---|
| LogisticRegression | |
| RandomForest | |

### System cost

| Model | Train time (ms) | Latency (ms) | Size (bytes) | Size (KB) | Peak RSS train (MB) | Peak RSS infer (MB) |
|---|---|---|---|---|---|---|
| LogisticRegression | | | | | | |
| RandomForest | | | | | | |

### Deployment fit

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| LogisticRegression | | | | |
| RandomForest | | | | |

## 4. Conclusions

1.
2.
3.
