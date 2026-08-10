# MLOps Training 2026/2027 — Task 1

## Get the Olist Data into a Database

This repository contains **Task 1** from an applied MLOps training project. The goal of this task is to take the Olist Brazilian e-commerce dataset, organize it as a relational database, validate the data model through SQL joins and integrity checks, and prepare a first machine-learning-ready view for studying late deliveries.

The work moves through a complete data-ingestion workflow: local dataset discovery, PostgreSQL connection, table creation, schema inspection, relationship testing, business-oriented SQL analysis, and SQLite export for convenient local access.

> The main deliverable is a reproducible foundation for the later stages of an end-to-end machine-learning and MLOps pipeline.

---

## Project Objectives

This task focuses on four practical objectives:

1. Load the Olist CSV files into PostgreSQL as separate relational tables.
2. Define and verify primary-key and foreign-key relationships between the tables.
3. Test the database with representative joins and exploratory business queries.
4. Create an ML-ready dataset that identifies whether a delivered order arrived late.

The notebook also exports the resulting tables to `olist.db`, allowing the data to be inspected locally through SQLite after the PostgreSQL workflow has been completed.

---

## Weekly Curriculum

The project is part of a 12-week applied MLOps program that progresses from reliable local machine-learning foundations to a complete production-ready system.

| Week | Phase | Dates | Topic | Focus Area |
|---|---|---|---|---|
| Week 1 | Local Foundations | 20–27 Jul 2026 | Leakage-proof ML Pipeline | Build robust pipelines that prevent data leakage and protect model integrity from the beginning. |
| Week 2 | Local Foundations | 27 Jul – 3 Aug 2026 | Deep Learning Pipeline | Extend the pipeline with neural-network architectures using PyTorch or TensorFlow. |
| Week 3 | Production APIs | 3–10 Aug 2026 | Production API | Create FastAPI or Flask endpoints to serve the model as a RESTful service. |
| Week 4 | Containerization | 10–17 Aug 2026 | Docker | Containerize the application to ensure consistency across development and production. |
| Week 5 | Data Pipelines | 17–24 Aug 2026 | ETL Pipeline | Design robust Extract–Transform–Load workflows using tools such as Airflow or Prefect. |
| Week 6 | Data Versioning | 24–31 Aug 2026 | Versioning | Implement DVC to version datasets and support reproducibility and collaboration. |
| Week 7 | Experiment Tracking | 31 Aug – 7 Sep 2026 | MLflow | Track experiments, parameters, metrics, and artifacts for reproducible development. |
| Week 8 | Distributed Training | 7–14 Sep 2026 | Distributed ML | Scale training across multiple GPUs or nodes using Ray, Horovod, or PyTorch Distributed. |
| Week 9 | Feature Store | 14–21 Sep 2026 | Feature Management | Build and manage a feature store for online and offline feature serving with Feast or Hopsworks. |
| Week 10 | Monitoring | 21–28 Sep 2026 | Monitoring | Implement model and data-drift monitoring using Prometheus, Grafana, or Evidently AI. |
| Week 11 | Continuous Retraining | 28 Sep – 5 Oct 2026 | Automation | Build automated retraining pipelines triggered by data drift or a schedule. |
| Week 12 | Infrastructure as Code | 5–12 Oct 2026 | Infrastructure | Define reproducible cloud infrastructure using Terraform or Pulumi. |
| Final | Capstone | 12–19 Oct 2026 | End-to-End ML System | Integrate all components into a complete, production-ready machine-learning system. |

The current repository work corresponds to the data-foundation stage: ingesting the Olist dataset, validating its relational structure, and preparing the data for the modeling and production stages that follow.

---

## Dataset Overview

The project uses the Olist Brazilian e-commerce dataset. It contains information about customers, orders, products, sellers, payments, reviews, shipping-related dates, and product categories.

| Dataset file | Main subject | Typical role in the analysis |
|---|---|---|
| `olist_customers_dataset.csv` | Customers and customer locations | Links orders to customer information |
| `olist_geolocation_dataset.csv` | Brazilian zip-code geolocation | Supports location-based analysis |
| `olist_orders_dataset.csv` | Order lifecycle and timestamps | Defines delivery status and dates |
| `olist_order_items_dataset.csv` | Products included in each order | Connects orders, products, and sellers |
| `olist_order_payments_dataset.csv` | Payment records | Describes payment methods and values |
| `olist_order_reviews_dataset.csv` | Customer reviews | Provides review scores and comments metadata |
| `olist_products_dataset.csv` | Product attributes | Adds product-level information |
| `olist_sellers_dataset.csv` | Seller records and locations | Connects items to sellers |
| `product_category_name_translation.csv` | Portuguese-to-English category names | Supports category interpretation |

The files are stored under `archive_2/` and are loaded programmatically by the notebook rather than being hard-coded one by one.

---

## Workflow

### 1. Environment and file discovery

The notebook imports the required Python libraries, defines the local dataset path, checks that the expected folder exists, and discovers the available CSV files.

### 2. PostgreSQL ingestion

A SQLAlchemy connection is configured using PostgreSQL connection settings. The notebook tests the connection before loading each CSV file into a PostgreSQL table. This makes the ingestion process easier to repeat when the dataset or database environment changes.

### 3. Schema and data validation

After ingestion, the notebook uses SQLAlchemy inspection to confirm that the tables exist and to report their row counts. It also prints table columns and SQL types so that the database structure can be reviewed before performing analytical queries.

### 4. Relationship and join tests

The notebook checks representative relationships between the main entities, including orders and customers, orders and order items, order items and products, order items and sellers, and orders and payments. These checks help confirm that the relational model can support downstream analysis.

### 5. Business-oriented analysis

The exploratory queries connect the database structure to the target business problem. The analysis includes order-status summaries, delivery-performance calculations, and a breakdown of late deliveries by customer state.

### 6. ML-ready view and SQLite export

The final query focuses on delivered orders and creates a binary `late` indicator. An order is marked as late when its actual delivery date is later than its estimated delivery date. The notebook then exports the PostgreSQL tables to `olist.db` for local SQLite usage.

---

## Database Relationships

The relationship definitions are documented in `Relationships.sql`. The main connections are summarized below.

| Child table | Key | Parent table | Relationship |
|---|---|---|---|
| `olist_orders` | `customer_id` | `olist_customers` | A customer may have multiple orders |
| `olist_order_items` | `order_id` | `olist_orders` | An order may contain multiple items |
| `olist_order_items` | `product_id` | `olist_products` | Each item refers to a product |
| `olist_order_items` | `seller_id` | `olist_sellers` | Each item refers to a seller |
| `olist_order_payments` | `order_id` | `olist_orders` | An order may have multiple payment records |
| `olist_order_reviews` | `order_id` | `olist_orders` | Reviews are associated with orders |

The SQL script also includes checks for primary keys and foreign keys. A foreign key is intentionally not created between `olist_products.product_category_name` and `product_category_name_translation.product_category_name`, because the translation table does not cover every product category present in the product table.

---

## Technology Stack

| Tool | Purpose |
|---|---|
| Python | Data ingestion and analysis |
| Jupyter Notebook | Interactive and reproducible execution |
| pandas | Reading CSV files and querying tabular data |
| SQLAlchemy | PostgreSQL connection and database inspection |
| PostgreSQL | Relational storage and SQL validation |
| SQLite | Local export and lightweight database access |
| SQL | Constraints, joins, validation, and business queries |

---

## Repository Structure

```text
Task1/
├── Task01.ipynb
├── Relationships.sql
├── olist.db
└── archive_2/
    ├── olist_customers_dataset.csv
    ├── olist_geolocation_dataset.csv
    ├── olist_order_items_dataset.csv
    ├── olist_order_payments_dataset.csv
    ├── olist_order_reviews_dataset.csv
    ├── olist_orders_dataset.csv
    ├── olist_products_dataset.csv
    ├── olist_sellers_dataset.csv
    └── product_category_name_translation.csv
```

`Task01.ipynb` contains the complete ingestion and validation workflow. `Relationships.sql` contains the relational constraints and verification queries. `olist.db` is the generated SQLite copy, while `archive_2/` contains the source CSV files.

---

## Prerequisites

Before running the notebook, make sure the following are available:

| Requirement | Purpose |
|---|---|
| Python 3.9 or newer | Runs the notebook code |
| Jupyter Notebook or JupyterLab | Executes `Task01.ipynb` |
| PostgreSQL | Stores the ingested tables |
| PostgreSQL credentials | Allows the notebook to connect to the database |
| Python packages | Supports data loading and SQL access |

The notebook expects the dataset folder to be available at the path assigned to `archive_path`. If the project is moved to another computer, update that path before execution.

A typical environment can be prepared with:

```bash
pip install pandas sqlalchemy psycopg2-binary jupyter
```

If the notebook uses additional packages in the local environment, install those packages before running all cells.

---

## How to Run the Project

### 1. Open the project directory

```bash
cd Task1
```

### 2. Start Jupyter

```bash
jupyter notebook
```

Open `Task01.ipynb` from the Jupyter interface.

### 3. Configure PostgreSQL

Update the database host, port, database name, username, and password in the connection-configuration cell. Confirm that the PostgreSQL server is running and that the selected database is accessible.

### 4. Set the dataset path

Point `archive_path` to the directory containing the nine Olist CSV files. The notebook will discover the files and load them into PostgreSQL.

### 5. Execute the notebook in order

Run the cells from top to bottom. The recommended sequence is setup, PostgreSQL connection, ingestion, schema inspection, join tests, business queries, ML-ready query, and SQLite export.

### 6. Apply the relationship script when required

After the tables have been loaded, run `Relationships.sql` in the target PostgreSQL database to define and verify the documented constraints. Review the verification queries before applying them to a production database.

---

## Validation Results

The notebook includes a final completion check for the ingested database. In the recorded run, the database contained nine tables and the delivery analysis returned the following results:

| Metric | Recorded value |
|---|---:|
| Delivered orders | 96,478 |
| Late deliveries | 7,826 |
| On-time deliveries | 88,644 |
| Late-delivery rate | 8.11% |

These values describe the execution captured in the notebook and may change if the source data, filtering conditions, or query logic is modified.

---

## ML-Ready Problem Definition

The first machine-learning-oriented target is a binary delivery label for delivered orders:

```text
late = 1  when order_delivered_customer_date > order_estimated_delivery_date
late = 0  otherwise
```

The current notebook prepares the target and validates the underlying dates. Future tasks can extend this view with features available before delivery, such as purchase timing, customer location, seller location, item count, product information, payment details, and shipping-related variables.

Care should be taken to avoid data leakage. Variables that are only known after delivery should not be used as predictive features for a model intended to estimate delivery risk before the order is completed.

---

## Notes and Limitations

The notebook is designed as a training task for database ingestion and validation, not as a complete production deployment. Database credentials should not be committed to GitHub; use environment variables or a local configuration file excluded through `.gitignore`.

The generated `olist.db` file is useful for local inspection, but PostgreSQL remains the primary relational database used by the ingestion workflow. When reproducing the project on another machine, the dataset path and database connection settings must be adapted to that environment.

---

## Next Steps

The natural continuation of this project is to build a reusable data-preparation pipeline, engineer leakage-safe delivery features, train and evaluate a binary classification model, track experiments, and expose the model through a reproducible inference workflow. These stages can then be connected to testing, packaging, monitoring, and deployment practices expected in an MLOps project.

---

## License and Data Attribution

This README documents the implementation contained in this repository. Review the terms and attribution requirements of the original Olist dataset before redistributing the data or using it outside the intended training context.
