import sqlite3

DB_PATH = "tracker.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# datasets table
cur.execute("""
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    row_count INTEGER,
    col_count INTEGER,
    schema_json TEXT,
    stats_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# models table
cur.execute("""
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    hyperparams TEXT,
    artifact_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# runs table (now includes tag + metadata_json)
cur.execute("""
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    metrics_json TEXT,
    notes TEXT,
    drift_report_path TEXT,
    tag TEXT,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (model_id) REFERENCES models(id)
)
""")

conn.commit()
conn.close()
print("✅ Database schema ready (with run tags + metadata_json)")
