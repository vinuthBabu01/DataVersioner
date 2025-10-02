import os
import json
import pickle
import sqlite3
import argparse
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from drift_detection import detect_drift, save_report

DB_PATH = "tracker.db"
ACCURACY_DROP_THRESHOLD = 0.05  # 5%


def load_dataset(identifier, by_version=True):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if by_version:
        cur.execute("SELECT id, file_path, version FROM datasets WHERE version=?", (identifier,))
    else:
        cur.execute("SELECT id, file_path, version FROM datasets WHERE id=?", (identifier,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"No dataset found with identifier {identifier}")
    dataset_id, file_path, version = row
    return dataset_id, file_path, version


def log_model(model_name, hyperparams, artifact_path):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO models (model_name, hyperparams, artifact_path)
        VALUES (?, ?, ?)
    """, (model_name, json.dumps(hyperparams), artifact_path))
    model_id = cur.lastrowid
    conn.commit()
    conn.close()
    return model_id


def log_run(dataset_id, model_id, metrics, notes="", drift_report_path=None, tag=None, metadata=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO runs (dataset_id, model_id, metrics_json, notes, drift_report_path, tag, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        dataset_id,
        model_id,
        json.dumps(metrics),
        notes,
        drift_report_path,
        tag,
        json.dumps(metadata) if metadata else None
    ))
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def train_logistic_regression(dataset_path, target_col):
    df = pd.read_csv(dataset_path)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if len(y.unique()) < 2:
        note = "Training skipped: single-class dataset"
        print(f"⚠️ {note}")
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "note": note
        }
        return None, metrics, note

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    return model, metrics, "Training successful"


def save_model(model, run_id):
    run_dir = f"models/run_{run_id:03d}"
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_version", type=str, required=True, help="Dataset version (e.g., v1, v2)")
    parser.add_argument("--target_col", type=str, required=True, help="Target column name")
    parser.add_argument("--baseline_run", type=int, help="Baseline run ID to compare accuracy against")
    parser.add_argument("--tag", type=str, help="Optional label for this run (e.g., baseline, exp1)")
    args = parser.parse_args()

    try:
        dataset_id, dataset_path, dataset_version = load_dataset(args.dataset_version, by_version=True)
    except Exception as e:
        print(f"❌ Failed to load dataset {args.dataset_version}: {e}")
        exit(1)

    # Load dataset into memory for metadata
    df = pd.read_csv(dataset_path)
    dataset_metadata = {
        "dataset_version": dataset_version,
        "row_count": df.shape[0],
        "col_count": df.shape[1],
        "features": [c for c in df.columns if c != args.target_col]
    }

    start_time = time.time()
    try:
        model, metrics, note = train_logistic_regression(dataset_path, args.target_col)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        exit(1)
    end_time = time.time()

    dataset_metadata["train_time_sec"] = round(end_time - start_time, 2)

    # Log model + run
    model_id = log_model("logistic_regression", {"max_iter": 500}, "")
    run_id = log_run(dataset_id, model_id, metrics, notes=note, tag=args.tag, metadata=dataset_metadata)

    if model is None:
        print(f"✅ Run {run_id} complete but training skipped. Metrics: {metrics}")
        exit(0)

    artifact_path = save_model(model, run_id)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE models SET artifact_path=? WHERE id=?", (artifact_path, model_id))
    conn.commit()
    conn.close()

    metrics_path = f"models/run_{run_id:03d}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Run {run_id} complete. Metrics: {metrics}")

    if args.baseline_run:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT dataset_id, metrics_json FROM runs WHERE id=?", (args.baseline_run,))
        baseline_row = cur.fetchone()
        conn.close()

        if baseline_row:
            baseline_dataset_id, baseline_metrics_json = baseline_row
            baseline_metrics = json.loads(baseline_metrics_json)
            acc_drop = baseline_metrics["accuracy"] - metrics["accuracy"]

            if acc_drop >= ACCURACY_DROP_THRESHOLD:
                print(f"⚠️ Accuracy dropped by {acc_drop:.2%}")
                print("🔎 Running drift detection...")

                drift_report = detect_drift(baseline_dataset_id, dataset_id, target_col=args.target_col, summary=True)
                report_path = f"models/run_{run_id:03d}/drift_report.txt"
                save_report(drift_report, report_path, summary=True)

                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("UPDATE runs SET drift_report_path=? WHERE id=?", (report_path, run_id))
                conn.commit()
                conn.close()

                print(f"📊 Drift report saved: {report_path}")
