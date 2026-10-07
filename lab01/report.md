# Lab 01: Environment and first system measurements
 
## 1. Goal
 
Set up a reproducible Python environment with pinned dependencies, train two baseline classifiers on the Breast Cancer Wisconsin dataset, measure their system cost (training time, inference latency, model size, peak memory) and decide which deployment tiers (Cloud, Edge, Mobile, TinyML) each model fits.
 
## 2. Method
 
- Environment: Python 3.11.8 in an isolated `.venv`, Windows, CPU only. Library versions are listed in `results/versions.txt`.
- Deviation from the manual: `pyarrow` is pinned to 15.0.2 instead of 16.1.0, because the required `mlflow==2.14.1` depends on `pyarrow<16` and the two specified versions cannot be installed together. Lab 1 does not use either package.
- Data: Breast Cancer Wisconsin (569 samples, 30 features), 70/30 stratified split, `random_state=42`. All seeds set to 42.
- Models: `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`, no feature scaling.
- Training time: 1 warm-up fit, then median of 5 fits.
- Inference latency: single test sample, 1 warm-up, then median of 100 `predict` calls.
- Model size: `joblib.dump` of the fitted model, file size in bytes and KB.
- Peak memory: `memory_profiler` peak process RSS during training and during 100 inference calls, plus the increase over the RSS measured just before the operation.
## 3. Results
 
### Accuracy
 
| Model | Test accuracy |
|---|---|
| LogisticRegression | 0.9415 |
| RandomForest | 0.9357 |
 
### System cost
 
| Model | Train time, median (ms) | Latency, median (ms) | Size (bytes) | Size (KB) | Peak RSS train (MB) | Peak RSS infer (MB) | Memory increase infer (MB) |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 428.696 | 0.0552 | 1055 | 1.03 | 264.84 | 264.84 | 0.0 |
| RandomForest | 121.327 | 2.2641 | 290889 | 284.07 | 266.04 | 266.04 | 0.0 |
 
### Deployment fit
 
Budgets: Cloud (memory ≥ 1 GB, latency ≤ 100 ms, size ≤ 500 MB), Edge (256–1024 MB, ≤ 50 ms, ≤ 50 MB), Mobile (64–256 MB, ≤ 20 ms, ≤ 10 MB), TinyML (≤ 256 KB, ≤ 10 ms, ≤ 100 KB).
 
| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| LogisticRegression | yes | yes | no as measured (process RSS 264.84 MB > 256 MB); yes for the model alone | no as measured; model alone fits (1.03 KB, 0.06 ms) but needs a non-Python runtime |
| RandomForest | yes | yes | no as measured (process RSS 266.04 MB > 256 MB); yes for the model alone | no: size 284.07 KB > 100 KB |
 
Full per-check table: `results/budget_fit.csv`.
 
Latency and size satisfy every tier except RandomForest size on TinyML. Memory is the deciding factor: about 265 MB is the footprint of the whole Python process (interpreter, NumPy, scikit-learn and PyTorch, which is imported for seeding), while the models themselves add almost nothing on top of it.
 
## 4. Conclusions
 
1. Logistic Regression is the better baseline for this dataset. It is slightly more accurate than Random Forest (0.9415 vs 0.9357), its single-sample latency is about 40 times lower (0.055 ms vs 2.26 ms) and the saved model is about 280 times smaller (1.03 KB vs 284 KB). The only metric where it loses is training time (429 ms vs 121 ms): without feature scaling the lbfgs solver hits the 1000-iteration limit, which is also why scikit-learn shows a ConvergenceWarning.
2. Memory, not the model, decides the deployment tier. Both models add almost nothing to memory, but the Python process with NumPy, scikit-learn and PyTorch already uses about 265 MB, which is just above the 256 MB Mobile limit. Both models fit Cloud and Edge as measured. To reach Mobile, the runtime has to get lighter (for example, not importing PyTorch or exporting the model to a lightweight format), not the model itself.
3. Only the Logistic Regression model is small enough for TinyML: 1.03 KB and 0.055 ms are well within 100 KB and 10 ms, while Random Forest is too large at 284 KB. However, the TinyML fit is theoretical, because a microcontroller cannot run Python; the 30 weights would need to be exported to C code. Also, all timings were measured on a laptop CPU, so the latency on real edge, mobile or TinyML hardware would be higher and should be measured on the target device.