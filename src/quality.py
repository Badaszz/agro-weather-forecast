import pandas as pd
import json
import os
from datetime import datetime, date, timedelta

QUALITY_LOG_PATH = "data/quality_results.json"

REQUIRED_COLUMNS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "et0_fao_evapotranspiration",
]

RANGE_RULES = {
    "temperature_2m_max":         (20.0, 45.0),
    "temperature_2m_min":         (15.0, 40.0),
    "precipitation_sum":          (0.0,  200.0),
    "windspeed_10m_max":          (0.0,  150.0),
    "et0_fao_evapotranspiration": (0.0,   20.0),
}

CLIP_RULES = {
    "temperature_2m_max":         (15.0,  55.0),
    "temperature_2m_min":         (10.0,  45.0),
    "precipitation_sum":          (0.0,  300.0),
    "windspeed_10m_max":          (0.0,  150.0),
    "et0_fao_evapotranspiration": (0.0,   20.0),
}


def load_raw_data(path: str = "data/raw_weather.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True)


def _check(passed: bool, detail: str) -> dict:
    return {"passed": bool(passed), "detail": str(detail)}


def run_quality_checks(df: pd.DataFrame) -> dict:
    results = {}

    # === CHECK 1: Minimum row count ===
    results["min_row_count_700"] = _check(
        len(df) >= 700,
        f"Found {len(df)} rows"
    )

    # === CHECK 2: Required columns present ===
    for col in REQUIRED_COLUMNS:
        exists = col in df.columns
        results[f"column_exists_{col}"] = _check(
            exists,
            f"Column '{col}' {'present' if exists else 'MISSING'}"
        )

    # === CHECK 3: Missing values below 10% per column ===
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            continue
        null_rate = df[col].isnull().mean()
        results[f"max_nulls_{col}"] = _check(
            null_rate <= 0.10,
            f"Null rate: {null_rate:.1%}"
        )

    # === CHECK 4: Value ranges (allow 5% outliers) ===
    for col, (lo, hi) in RANGE_RULES.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).mean()
        results[f"range_{col}"] = _check(
            out_of_range <= 0.05,
            f"Out of range: {out_of_range:.1%} | "
            f"Observed: {df[col].min():.1f} – {df[col].max():.1f}"
        )

    # === CHECK 5: No duplicate dates ===
    dupes = df.index.duplicated().sum()
    results["no_duplicate_dates"] = _check(
        dupes == 0,
        f"{'No duplicates' if dupes == 0 else f'{dupes} duplicate date(s) found'}"
    )

    # === CHECK 6: Date continuity (no gaps > 1 day) ===
    if len(df) > 1:
        diffs   = df.index.to_series().diff().dropna()
        max_gap = diffs.max().days
        results["no_date_gaps"] = _check(
            max_gap <= 1,
            f"Max gap between dates: {max_gap} day(s)"
        )
    else:
        results["no_date_gaps"] = _check(False, "Not enough rows to check continuity")

    # === SUMMARY ===
    total  = len(results)
    passed = sum(1 for v in results.values() if v["passed"])

    return {
        "run_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total":   total,
        "passed":  passed,
        "failed":  total - passed,
        "overall": "PASS" if total == passed else "FAIL",
        "checks":  results,
    }


def save_quality_results(summary: dict, path: str = QUALITY_LOG_PATH):
    os.makedirs("data", exist_ok=True)
    history = []
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.append(summary)
    history = history[-24:]  # keep last 24 runs
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"💾 Quality results saved to {path}")


def print_quality_report(summary: dict):
    icon = "✅" if summary["overall"] == "PASS" else "❌"
    print(f"\n{'='*56}")
    print(f"  {icon} DATA QUALITY REPORT — {summary['run_at']}")
    print(f"{'='*56}")
    print(f"  Overall : {summary['overall']}")
    print(f"  Passed  : {summary['passed']} / {summary['total']}")
    print(f"  Failed  : {summary['failed']} / {summary['total']}")
    print(f"{'-'*56}")
    for name, result in summary["checks"].items():
        status = "✅" if result["passed"] else "❌"
        label  = name.replace("_", " ")
        print(f"  {status}  {label:<38} {result['detail']}")
    print(f"{'='*56}\n")


def repair_raw_data(path: str = "data/raw_weather.csv") -> bool:
    df       = pd.read_csv(path, index_col="date", parse_dates=True)
    repaired = False
    report   = []

    # Fix 1: Duplicate dates
    dupes = df.index.duplicated(keep="last").sum()
    if dupes > 0:
        df = df[~df.index.duplicated(keep="last")]
        report.append(f"Removed {dupes} duplicate date(s)")
        repaired = True

    # Fix 2: Missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        df = df.interpolate(method="time").ffill().bfill()
        report.append(f"Interpolated nulls: {null_counts[null_counts > 0].to_dict()}")
        repaired = True

    # Fix 3: Date gaps — reindex and interpolate
    if len(df) > 1:
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        if len(full_range) > len(df):
            df = df.reindex(full_range)
            df.index.name = "date"
            df = df.interpolate(method="time").ffill().bfill()
            report.append(f"Filled {len(full_range) - len(df)} date gap(s)")
            repaired = True

    # Fix 4: Row count too low — re-fetch
    if len(df) < 700:
        report.append(f"Row count too low ({len(df)}) — re-fetching from API...")
        try:
            from src.ingestion import fetch_weather_data, save_raw_data
            df_fresh = fetch_weather_data(
                start_date=(date.today() - timedelta(days=800)).strftime("%Y-%m-%d"),
                end_date=(date.today() - timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            save_raw_data(df_fresh, path)
            report.append(f"Re-fetched {len(df_fresh)} rows successfully")
            repaired = True
            df = pd.read_csv(path, index_col="date", parse_dates=True)
        except Exception as e:
            report.append(f"Re-fetch failed: {e}")

    # Fix 5: Clip corrupt extreme values
    for col, (lo, hi) in CLIP_RULES.items():
        if col in df.columns:
            bad = ((df[col] < lo) | (df[col] > hi)).sum()
            if bad > 0:
                df[col] = df[col].clip(lower=lo, upper=hi)
                report.append(f"Clipped {bad} corrupt value(s) in '{col}'")
                repaired = True

    if repaired:
        df.to_csv(path)
        print("🔧 Data repair applied:")
        for line in report:
            print(f"   → {line}")
    else:
        print("ℹ️  No auto-repairs needed.")

    return repaired


def run_pipeline_quality_gate(path: str = "data/raw_weather.csv") -> bool:
    print("🔍 Running data quality checks...")
    df      = load_raw_data(path)
    summary = run_quality_checks(df)
    print_quality_report(summary)
    save_quality_results(summary)

    if summary["overall"] == "PASS":
        print("✅ Quality gate PASSED — proceeding to retraining.")
        return True

    print("⚠️  Quality gate FAILED — attempting auto-repair...")
    repaired = repair_raw_data(path)

    if repaired:
        print("\n🔍 Re-running quality checks after repair...")
        df      = load_raw_data(path)
        summary = run_quality_checks(df)
        print_quality_report(summary)
        save_quality_results(summary)

        if summary["overall"] == "PASS":
            print("✅ Quality gate PASSED after repair — proceeding to retraining.")
            return True

    failed = [k for k, v in summary["checks"].items() if not v["passed"]]
    print("🚨 Quality gate FAILED after repair attempt.")
    print(f"   Failed checks : {failed}")
    print("   Existing model kept. No retraining this month.")
    return False


if __name__ == "__main__":
    passed = run_pipeline_quality_gate()
    exit(0 if passed else 1)