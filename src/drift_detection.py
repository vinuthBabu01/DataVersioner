import pandas as pd
import sqlite3
from scipy.stats import ks_2samp, chi2_contingency
import argparse
import json
import os

DB_PATH = "tracker.db"

def load_dataset_from_db(identifier, by_version=False):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if by_version:
        cur.execute("SELECT file_path FROM datasets WHERE version=?", (identifier,))
    else:
        cur.execute("SELECT file_path FROM datasets WHERE id=?", (identifier,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"No dataset found with identifier {identifier}")
    return pd.read_csv(row[0])

def ks_test(feature, df1, df2):
    stat, pval = ks_2samp(df1[feature].dropna(), df2[feature].dropna())
    return {"ks_stat": float(stat), "p_value": float(pval)}

def chi_square_test(feature, df1, df2):
    counts1 = df1[feature].value_counts().sort_index()
    counts2 = df2[feature].value_counts().sort_index()
    df = pd.DataFrame({"v1": counts1, "v2": counts2}).fillna(0)
    stat, pval, _, _ = chi2_contingency(df)
    return {"chi2_stat": float(stat), "p_value": float(pval)}

def detect_drift(v1, v2, target_col=None, alpha=0.05, summary=False, by_version=False):
    df1 = load_dataset_from_db(v1, by_version=by_version)
    df2 = load_dataset_from_db(v2, by_version=by_version)

    drift_report = {}
    human_summary = []

    for col in df1.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df1[col]):
            res = ks_test(col, df1, df2)
            drift_report[col] = res
            if summary:
                if res["p_value"] < alpha:
                    human_summary.append(
                        f"📈 Feature '{col}' drifted (KS={res['ks_stat']:.2f}, p={res['p_value']:.3g})"
                    )
                else:
                    human_summary.append(
                        f"✅ Feature '{col}' stable (KS={res['ks_stat']:.2f}, p={res['p_value']:.3g})"
                    )
        else:
            res = chi_square_test(col, df1, df2)
            drift_report[col] = res
            if summary:
                if res["p_value"] < alpha:
                    human_summary.append(
                        f"🔢 Feature '{col}' categorical drift (Chi2={res['chi2_stat']:.2f}, p={res['p_value']:.3g})"
                    )
                else:
                    human_summary.append(
                        f"✅ Feature '{col}' categorical stable (Chi2={res['chi2_stat']:.2f}, p={res['p_value']:.3g})"
                    )

    if target_col and target_col in df1.columns:
        dist1 = df1[target_col].value_counts(normalize=True).to_dict()
        dist2 = df2[target_col].value_counts(normalize=True).to_dict()
        drift_report[target_col] = {"v1_distribution": dist1, "v2_distribution": dist2}
        if summary:
            human_summary.append(f"🎯 Target drift: v1={dist1}, v2={dist2}")

    return human_summary if summary else drift_report

def save_report(report, path, summary=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if summary:
            f.write("\n".join(report))
        else:
            json.dump(report, f, indent=4)
    print(f"✅ Drift report saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", required=True, help="Dataset ID or version for v1")
    parser.add_argument("--v2", required=True, help="Dataset ID or version for v2")
    parser.add_argument("--target_col", type=str, default=None, help="Target column name")
    parser.add_argument("--summary", action="store_true", help="Print human-readable summary instead of JSON")
    parser.add_argument("--by_version", action="store_true", help="Use dataset versions (v1, v2) instead of IDs")
    parser.add_argument("--save", type=str, help="Optional file path to save the drift report")
    args = parser.parse_args()

    report = detect_drift(
        args.v1,
        args.v2,
        target_col=args.target_col,
        summary=args.summary,
        by_version=args.by_version
    )

    if args.summary:
        print("\n".join(report))
    else:
        print(json.dumps(report, indent=4))

    if args.save:
        save_report(report, args.save, summary=args.summary)
