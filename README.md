# DataVersioner

DataVersioner is a lightweight tool for **dataset and model versioning** with built-in **drift detection** and **explanation engine**.  
It helps you understand **why a machine learning model’s performance changes** across different training runs, especially when accuracy drops.

---

## ✨ Features
- 🔹 Track **dataset versions** with hashes, schema, and metadata.
- 🔹 Store **model runs** with artifacts, hyperparameters, and metrics.
- 🔹 Detect **data drift** (distribution shifts, class imbalance).
- 🔹 Monitor **model drift** (performance degradation).
- 🔹 Generate **explanations** for accuracy drops in natural language.
- 🔹 Works with **any dataset size** — from thousands to millions of rows.

---

## 📂 Project Structure
DataVersioner/
│
├── data/ # Dataset versions (raw + metadata)
├── models/ # Model artifacts + metrics
├── src/ # Source code (ingest, train, drift, explain, report)
├── notebooks/ # Jupyter notebooks for testing
├── scripts/ # CLI helpers
├── tracker.db # SQLite DB for metadata tracking
├── requirements.txt # Python dependencies
└── README.md # Project documentation