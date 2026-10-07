import pandas as pd

from common import RESULTS

INF = float("inf")
BUDGETS = {
    "Cloud": {"mem_mb": INF, "latency_ms": 100, "size_kb": 500 * 1024},
    "Edge": {"mem_mb": 1024, "latency_ms": 50, "size_kb": 50 * 1024},
    "Mobile": {"mem_mb": 256, "latency_ms": 20, "size_kb": 10 * 1024},
    "TinyML": {"mem_mb": 0.25, "latency_ms": 10, "size_kb": 100},
}


def mark(ok):
    return "yes" if ok else "no"


def main():
    m = pd.read_csv(RESULTS / "measurements.csv")
    rows = []
    for _, r in m.iterrows():
        for tier, b in BUDGETS.items():
            lat_ok = r["latency_ms_median"] <= b["latency_ms"]
            size_ok = r["size_kb"] <= b["size_kb"]
            mem_proc_ok = r["infer_peak_rss_mb"] <= b["mem_mb"]
            mem_model_ok = r["infer_delta_mb"] <= b["mem_mb"]
            rows.append({
                "model": r["model"],
                "tier": tier,
                "latency_ok": mark(lat_ok),
                "size_ok": mark(size_ok),
                "memory_ok_process": mark(mem_proc_ok),
                "memory_ok_model_only": mark(mem_model_ok),
                "fits_as_measured": mark(lat_ok and size_ok and mem_proc_ok),
                "fits_model_only": mark(lat_ok and size_ok and mem_model_ok),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "budget_fit.csv", index=False)

    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    (RESULTS / "budget_fit.md").write_text("\n".join([header, sep, *body]) + "\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
