import sqlite3
import json
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt

DB_PATH = "tracker.db"


def list_datasets():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, version, row_count, col_count, created_at FROM datasets")
    rows = cur.fetchall()
    conn.close()
    headers = ["ID", "Version", "Rows", "Cols", "Created At"]
    return tabulate(rows, headers=headers, tablefmt="pretty")


def list_runs():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT r.id, d.version, m.model_name, r.metrics_json, r.notes,
               r.drift_report_path, r.tag, r.metadata_json, r.created_at
        FROM runs r
        JOIN datasets d ON r.dataset_id = d.id
        JOIN models m ON r.model_id = m.id
        ORDER BY r.id ASC
    """)
    rows = cur.fetchall()
    conn.close()

    results = []
    for run_id, dataset_version, model_name, metrics_json, notes, drift_report_path, tag, metadata_json, created_at in rows:
        metrics = json.loads(metrics_json)
        accuracy = metrics.get("accuracy", None)
        metadata = json.loads(metadata_json) if metadata_json else {}
        row_count = metadata.get("row_count", "-")
        col_count = metadata.get("col_count", "-")
        train_time = metadata.get("train_time_sec", "-")

        results.append({
            "run_id": run_id,
            "dataset_version": dataset_version,
            "model_name": model_name,
            "accuracy": accuracy,
            "metrics": metrics,
            "notes": notes if notes else "-",
            "drift_report_path": drift_report_path if drift_report_path else "-",
            "tag": tag if tag else "-",
            "shape": f"{row_count}x{col_count}",
            "train_time": train_time,
            "created_at": created_at
        })
    return results


def display_runs_table(runs):
    table = []
    for run in runs:
        table.append([
            run["run_id"],
            run["dataset_version"],
            run["model_name"],
            run["accuracy"],
            run["notes"],
            run["drift_report_path"],
            run["tag"],
            run["shape"],
            run["train_time"],
            run["created_at"]
        ])
    headers = [
        "Run ID", "Dataset", "Model", "Accuracy",
        "Notes", "Drift Report", "Tag",
        "Shape", "Train Time (s)", "Created At"
    ]
    return tabulate(table, headers=headers, tablefmt="pretty")


def plot_metric(runs, metric="accuracy"):
    run_ids = [r["run_id"] for r in runs]
    values = [r["metrics"].get(metric, None) for r in runs]
    tags = [r["tag"] for r in runs]

    if all(v is None for v in values):
        print(f"❌ Metric '{metric}' not found in runs.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(run_ids, values, marker="o", linestyle="-", label=metric.capitalize())

    # Add labels (tags or run IDs) on each point
    for i, (x, y) in enumerate(zip(run_ids, values)):
        label = tags[i] if tags[i] != "-" else str(x)
        plt.text(x, y, label, fontsize=9, ha="right", va="bottom")

    plt.title(f"{metric.capitalize()} over Runs")
    plt.xlabel("Run ID")
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, help="Metric to plot (e.g., accuracy, precision, recall, f1)")
    args = parser.parse_args()

    runs = list_runs()

    if args.plot:
        plot_metric(runs, args.plot)
    else:
        print("📊 Datasets:\n")
        print(list_datasets())
        print("\n📊 Runs:\n")
        print(display_runs_table(runs))
