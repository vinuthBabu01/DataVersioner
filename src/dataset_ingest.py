import os
import hashlib
import pandas as pd
import json
import sqlite3
import argparse

DB_PATH = "tracker.db"


def compute_file_hash(file_path, chunk_size=8192):
    """Compute SHA256 hash of file for uniqueness check"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_metadata(file_path, fmt="csv", row_start=None, row_end=None, sample_size=1_000_000):
    """Generate dataset metadata for CSV/Parquet with optional row slicing"""
    metadata = {}
    metadata["file_hash"] = compute_file_hash(file_path)

    if fmt == "csv":
        # Handle chunked reading for large CSV
        df_iter = pd.read_csv(file_path, chunksize=sample_size)
        total_rows, dtypes, total_nulls, stats = 0, {}, {}, {}
        for i, chunk in enumerate(df_iter):
            # Apply row slicing if requested (only on first chunk)
            if row_start is not None or row_end is not None:
                chunk = chunk.iloc[row_start:row_end]
            total_rows += len(chunk)
            nulls = chunk.isnull().sum().to_dict()
            for col, val in nulls.items():
                total_nulls[col] = total_nulls.get(col, 0) + val
            if i == 0:  # collect schema/stats only once
                dtypes = chunk.dtypes.astype(str).to_dict()
                stats = chunk.describe(include="all").to_dict()
            if row_start is not None or row_end is not None:
                break  # if row slicing, stop after first chunk
        metadata["row_count"] = total_rows
        metadata["col_count"] = len(dtypes)
        metadata["dtypes"] = dtypes
        metadata["null_counts"] = total_nulls
        metadata["sample_stats"] = stats

    elif fmt == "parquet":
        df = pd.read_parquet(file_path)
        if row_start is not None or row_end is not None:
            df = df.iloc[row_start:row_end]
        metadata["row_count"] = len(df)
        metadata["col_count"] = len(df.columns)
        metadata["dtypes"] = df.dtypes.astype(str).to_dict()
        metadata["null_counts"] = df.isnull().sum().to_dict()
        metadata["sample_stats"] = df.describe(include="all").to_dict()

    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return metadata


def save_metadata(metadata, version_dir):
    os.makedirs(version_dir, exist_ok=True)
    with open(os.path.join(version_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def log_to_db(version, file_path, metadata, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO datasets (version, file_path, file_hash, row_count, col_count, schema_json, stats_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            version,
            file_path,
            metadata["file_hash"],
            metadata["row_count"],
            metadata["col_count"],
            json.dumps(metadata["dtypes"]),
            json.dumps(metadata["sample_stats"])
        ))
        conn.commit()
        print(f"✅ Dataset {version} logged in DB")
    except sqlite3.IntegrityError:
        print(f"⚠️ Dataset with hash {metadata['file_hash']} already exists. Skipping insert.")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--fmt", type=str, default=None, help="File format: csv or parquet (auto-detect if not set)")
    parser.add_argument("--version", type=str, required=True, help="Dataset version label (e.g., v1, v2)")
    parser.add_argument("--rows", nargs=2, type=int, help="Optional row range to ingest, e.g. --rows 0 1000000")
    args = parser.parse_args()

    # Auto-detect format if not provided
    fmt = args.fmt
    if fmt is None:
        if args.file.endswith(".csv"):
            fmt = "csv"
        elif args.file.endswith(".parquet"):
            fmt = "parquet"
        else:
            raise ValueError("Unknown file type. Please specify --fmt csv or parquet.")

    row_start, row_end = None, None
    if args.rows:
        row_start, row_end = args.rows

    metadata = generate_metadata(args.file, fmt=fmt, row_start=row_start, row_end=row_end)
    save_metadata(metadata, f"data/{args.version}")
    log_to_db(args.version, args.file, metadata)
