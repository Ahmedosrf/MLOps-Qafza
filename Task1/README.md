# Task 1 — Get the Data Into a Database

This folder contains the completed foundation for the Olist MLOps training project. The task loads the raw Olist CSV files into a local PostgreSQL database, verifies the schema and row counts, tests the principal relationships, and defines the late-delivery classification problem for later tasks.

## Folder structure

```text
Task1/
├── notebooks/
│   └── Task01.ipynb              # Main executed notebook for Task 1
├── sql/
│   └── Relationships.sql         # Optional primary-key and foreign-key constraints
├── data/                          # Raw Olist CSV files
├── database/
│   └── olist.db                  # Local SQLite copy of the loaded tables
├── src/
│   ├── __init__.py
│   └── config.py                  # Reusable paths and database settings
├── .env.example                   # Safe configuration template
├── README.md
└── docs/
    └── notebook_summary.txt      # Extracted notebook summary and cell reference
```

## Main notebook

Run `notebooks/Task01.ipynb` from top to bottom. It performs the following steps:

1. Checks the local CSV folder and lists the available files.
2. Connects to PostgreSQL using environment variables.
3. Loads the CSV files into relational tables.
4. Verifies the tables, row counts, and schema.
5. Tests joins between orders, customers, items, products, sellers, and payments.
6. Calculates the delivered, late, and on-time order summary.
7. Runs the final Task 1 completion check.

The notebook reads raw CSV files from the configured data directory and expects the PostgreSQL database `olist` to be available on `localhost:5432` with the default user `postgres`. Paths and connection settings are loaded through `src/config.py`, so the notebook does not depend on a fixed computer path or a password embedded in source code. Set the password without writing it into the notebook:

```powershell
$env:OLIST_DB_PASSWORD = "your_password"
```

Optional variables are `MLOPS_PROJECT_ROOT`, `OLIST_DATA_DIR`, `OLIST_DB_HOST`, `OLIST_DB_PORT`, `OLIST_DB_NAME`, and `OLIST_DB_USER`. See `.env.example` for the names and default values. The real `.env` file must remain local.

## Relationships

After the raw tables have been loaded and verified, `sql/Relationships.sql` may be executed in PostgreSQL to add primary keys and foreign keys. The category translation table is intentionally not assigned a foreign key because it does not contain every product category found in the products table.

## Scope

Task 1 is a data-ingestion and database-foundation task. Full EDA, cleaning, feature engineering, model training, and evaluation belong to later tasks. When building the future order-level ML table, aggregate one-to-many tables by `order_id` before joining them, and avoid using future information such as actual delivery dates or reviews as prediction features.

## Official dataset

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
