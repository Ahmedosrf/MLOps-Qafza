# MLOps-Qafza — From Data Foundations to Production ML

[![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![SQLite](https://img.shields.io/badge/Export-SQLite-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![SQL](https://img.shields.io/badge/Queries-SQL-4479A1?logo=mysql&logoColor=white)](#technology-stack)

> **MLOps-Qafza** is an applied MLOps training project that progresses from reliable data foundations to reproducible machine-learning workflows and, eventually, a complete production-ready ML system.

## Project Overview

This repository documents a practical MLOps learning journey based on the Olist Brazilian e-commerce dataset. The project begins with raw relational data and gradually develops into a complete machine-learning and production workflow.

The learning path covers database ingestion, relational validation, leakage-aware machine-learning pipelines, deep learning, production APIs, containerization, data pipelines, data versioning, experiment tracking, distributed training, feature stores, monitoring, continuous retraining, infrastructure as code, and an end-to-end capstone system.

The repository is organized by task. Each task has its own README containing its objectives, implementation details, execution instructions, outputs, validation results, and limitations.

## Project Objectives

The project has four complementary objectives:

1. Build a reliable foundation for storing and validating real-world data.
2. Develop reproducible machine-learning workflows that avoid data leakage.
3. Progress from local experimentation toward production-oriented MLOps practices.
4. Document every stage clearly enough to support reproducibility, learning, and collaboration.

## Training Roadmap

The project is part of a 12-week applied MLOps program. It progresses from local machine-learning foundations to production engineering and concludes with a complete end-to-end system.

## Weekly Curriculum

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

## Current Tasks

| Task | Description | Documentation | Status |
|---|---|---|---|
| Task 1 | Ingest the Olist data into PostgreSQL and validate its relational structure. | [Task 1 README](./Task1/README.md) | Completed |
| Task 2 | Build a leakage-aware machine-learning pipeline for late-delivery classification. | [Task 2 README](./Task2/README.md) | Completed |
| Task 3 | Build a production API, containerize the inference stack, and add production foundations. | [Task 3 README](./Task3/README.md) | Completed |
| Task 4 | Containerize the application using Docker. | [Task 4 README](./Task4/README.md) | Planned / foundations in Task 3 |
| Task 5 | Build reproducible ETL and data-pipeline workflows. | [Task 5 README](./Task5/README.md) | Planned |
| Task 6 | Add dataset versioning using DVC. | [Task 6 README](./Task6/README.md) | Planned |
| Task 7 | Add experiment tracking using MLflow. | [Task 7 README](./Task7/README.md) | Planned |
| Task 8 | Explore distributed machine learning. | [Task 8 README](./Task8/README.md) | Planned |
| Task 9 | Add feature-store capabilities. | [Task 9 README](./Task9/README.md) | Planned |
| Task 10 | Add data and model monitoring. | [Task 10 README](./Task10/README.md) | Planned |
| Task 11 | Implement continuous retraining. | [Task 11 README](./Task11/README.md) | Planned |
| Task 12 | Implement infrastructure as code and deployment practices. | [Task 12 README](./Task12/README.md) | Planned |

Task 1 and Task 2 represent the **Local Foundations** stage. Task 3 adds the first production-oriented inference service, container stack, artifact handoff, validation, observability, and CI foundations.

## Problem Definition

The initial machine-learning problem is binary classification of delivery outcomes for delivered orders:

```text
late = 1      when the actual delivery date is after the estimated delivery date
on_time = 0   when the actual delivery date is on or before the estimated date
```

The objective is to identify orders that are likely to be delivered late while avoiding data leakage. The detailed target definition, feature policy, splitting strategy, and evaluation results are documented in [Task 2](./Task2/README.md).

## Dataset Overview

The project uses the Olist Brazilian e-commerce dataset. It contains information about customers, orders, products, sellers, payments, reviews, shipping-related dates, product categories, and geographical information.

| Dataset area | Examples | Role |
|---|---|---|
| Customers | Customer IDs, cities, states, postal codes | Links orders to customer information |
| Orders | Statuses and timestamps | Defines the order lifecycle and delivery outcome |
| Order items | Products, sellers, prices, freight | Describes order contents and commercial value |
| Payments | Payment methods, values, installments | Describes payment behavior |
| Reviews | Scores and review metadata | Supports post-order analysis |
| Products | Categories, weights, dimensions | Adds product-level attributes |
| Sellers | Seller IDs and locations | Connects items to seller information |
| Geolocation | Postal codes and coordinates | Supports geographical analysis |

Detailed table definitions, database relationships, and ingestion instructions are documented in [Task 1](./Task1/README.md).

## Project Workflow

```text
Raw Olist CSV files
        ↓
Relational PostgreSQL database
        ↓
Schema and relationship validation
        ↓
Order-level joined table
        ↓
Late-delivery label creation
        ↓
Train / validation / test split
        ↓
Training-only EDA
        ↓
Leakage-safe feature engineering
        ↓
Model training and evaluation
        ↓
Production API and MLOps stages
```

## Technology Stack

| Area | Technologies | Purpose |
|---|---|---|
| Programming | Python | Data ingestion, analysis, and machine learning |
| Data processing | pandas, NumPy | Tabular data processing and numerical operations |
| Interactive analysis | Jupyter Notebook | Reproducible, cell-based development |
| Relational database | PostgreSQL | Primary relational data storage |
| Database access | SQLAlchemy, psycopg2 | Database connections and inspection |
| Query language | SQL | Joins, constraints, validation, and analysis |
| Local export | SQLite | Lightweight local inspection |
| Machine learning | scikit-learn | Feature processing, modeling, and evaluation |
| Data formats | CSV, Parquet, NumPy arrays | Input and intermediate data storage |
| APIs | FastAPI, Pydantic, Uvicorn | Typed model-serving API and request validation |
| Containers | Docker | Reproducible runtime environments |
| Data versioning | DVC | Versioning datasets and data pipelines |
| Experiment tracking | MLflow | Tracking parameters, metrics, models, and artifacts |
| Monitoring | Prometheus, Grafana, Evidently AI | Monitoring data and model behavior |
| Infrastructure | Terraform or Pulumi | Reproducible infrastructure definition |

## Repository Structure

```text
MLOps-Qafza/
├── README.md                              ← General project overview and roadmap
├── .gitignore                             ← Ignored local, generated, and sensitive files
├── .gitattributes                          ← Git file-handling configuration
│
├── Task1/
│   ├── README.md                          ← Task 1 documentation
│   ├── docker-compose.yml                  ← Optional local PostgreSQL service
│   ├── notebooks/
│   │   └── Task01.ipynb                   ← Database ingestion and validation notebook
│   ├── sql/
│   │   └── Relationships.sql              ← Database relationships and integrity checks
│   ├── src/
│   │   └── config.py                       ← Portable data and database configuration
│   ├── data/                               ← Local Olist CSV files; normally not committed
│   ├── database/
│   │   └── olist.db                        ← Local SQLite export; normally not committed
│   └── docs/
│       └── notebook_summary.txt            ← Supporting notebook summary
│
├── Task2/
│   ├── README.md                          ← Task 2 documentation
│   ├── notebooks/
│   │   ├── 01_read_and_join_tables.ipynb
│   │   ├── 02_create_labels.ipynb
│   │   ├── 03_split_data.ipynb
│   │   ├── 04_eda.ipynb
│   │   ├── 05_feature_engineering.ipynb
│   │   └── 06_train_tune_evaluate.ipynb
│   ├── src/
│   │   └── config.py                       ← Portable project-root configuration
│   └── artifacts/                          ← Generated outputs; normally not committed
│       ├── tables/
│       ├── features/
│       ├── reports/
│       ├── charts/
│       └── models/
│
├── Task3/                                  ← Production inference and API task
│   ├── README.md
│   ├── app/
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   ├── .env.example
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── requirements-dev.txt
├── .dockerignore                           ← Docker build-context exclusions
├── .github/workflows/ci.yml                 ← Automated Task 3 tests
├── Task4/                                  ← Future Docker task
│   └── README.md
├── Task5/                                  ← Future ETL task
│   └── README.md
├── Task6/                                  ← Future data-versioning task
│   └── README.md
├── Task7/                                  ← Future experiment-tracking task
│   └── README.md
├── Task8/                                  ← Future distributed-training task
│   └── README.md
├── Task9/                                  ← Future feature-store task
│   └── README.md
├── Task10/                                 ← Future monitoring task
│   └── README.md
├── Task11/                                 ← Future continuous-retraining task
│   └── README.md
├── Task12/                                 ← Future infrastructure-as-code task
│   └── README.md
└── docs/                                   ← Optional repository-wide documentation
```

The root README provides the project map. Each task README provides the detailed documentation for that task. This separation keeps the repository readable while preserving the full history and technical details of every stage.

## Task Documentation Map

| Documentation file | Scope |
|---|---|
| `README.md` | Project-wide overview, curriculum, roadmap, status, and navigation |
| `Task1/README.md` | PostgreSQL ingestion, relational validation, SQL relationships, and database execution |
| `Task2/README.md` | Labels, data splitting, EDA, feature engineering, model training, and evaluation |
| `Task3/README.md` | FastAPI inference, Docker Compose, MLflow, DVC, validation, metrics, and CI |
| `TaskN/README.md` | Detailed documentation for each future task |

## Reproducibility Principles

Each task is separated into clear stages with documented inputs and outputs. Data ingestion, target creation, splitting, EDA, feature engineering, model training, and evaluation should not be mixed into one unstructured notebook.

The project should be reproducible from a clean environment whenever possible. Paths should be configurable, generated outputs should be clearly named, and each stage should record the assumptions and validation results that support it.

## Security and Data Management

Database credentials must be supplied through environment variables or a local configuration file excluded by `.gitignore`. Passwords, API keys, private connection strings, and personal credentials must never be committed to GitHub.

Large raw datasets, local database files, generated model files, and temporary artifacts should normally remain outside version control unless the training requirements explicitly request them. The README files document the expected paths even when the actual data is kept locally.

## Current Progress

### Task 1 — Completed

Task 1 established the relational data foundation by loading the Olist dataset into PostgreSQL, validating the schema and joins, testing database relationships, and defining the late-delivery problem. Its notebook now reads paths and database settings from reusable configuration instead of relying on a fixed local path or an embedded password.

The task includes a complete notebook, a SQL relationship script, a local SQLite export, and documentation of the database workflow.

### Task 2 — Completed

Task 2 extended the foundation into a reproducible machine-learning pipeline. It creates the `is_late` target, uses a temporal train/validation/test split, performs EDA on training data, builds leakage-aware features, trains a baseline and classifier, and evaluates the result using multiple metrics rather than accuracy alone. It also produces a versioned inference handoff containing the fitted preprocessor, feature list, selected threshold, model, and model metadata for Task 3.

The Task 2 pipeline stores intermediate outputs in separate artifact categories: tables, features, reports, charts, and models.

### Task 3 — Completed production foundations

Task 3 serves the fitted Task 2 bundle through FastAPI without retraining. It includes request validation, leakage-safe prediction-time feature construction, model metadata, batch inference, Prometheus metrics, structured logs, Docker Compose, MLflow with PostgreSQL and MinIO, DVC handoff metadata, Great Expectations integration, and GitHub Actions quality checks. The model binary remains outside Git and must be supplied through the documented local artifact, DVC remote, or artifact-store workflow.

## Validation Philosophy

The project does not treat accuracy as the only indicator of model quality. Because late deliveries represent a smaller class, evaluation should consider recall, precision, F1-score, ROC-AUC or PR-AUC, and the confusion matrix.

A model that detects nearly every late order but produces too many false alarms is not equivalent to a model that provides balanced and operationally useful predictions. The evaluation objective must therefore be connected to the actual business cost of missed late orders and unnecessary alerts.

## Future Roadmap

The next stages will extend the current foundations toward deep learning, ETL orchestration, distributed training, feature stores, monitoring, automated retraining, infrastructure as code, and the final capstone system.

The intended long-term workflow is:

```text
Reliable data foundation
        ↓
Reproducible ML pipeline
        ↓
Deep learning extensions
        ↓
Production API
        ↓
Containerized service
        ↓
Versioned data and experiments
        ↓
Monitoring and automated retraining
        ↓
Infrastructure as code
        ↓
End-to-end production ML system
```

## Prerequisites

The exact prerequisites depend on the task, but the foundation currently uses:

| Requirement | Purpose |
|---|---|
| Python 3.11–3.13 | Runs the data, machine-learning, and inference code |
| Jupyter Notebook or JupyterLab | Executes the notebooks |
| PostgreSQL | Stores the relational Olist tables |
| PostgreSQL credentials | Allows local database access |
| pandas and NumPy | Data manipulation and numerical processing |
| SQLAlchemy and psycopg2-binary | PostgreSQL connectivity |
| pyarrow | Reading and writing Parquet files |
| scikit-learn | Feature processing and machine learning |

A typical foundation environment can be prepared with:

```bash
pip install pandas numpy sqlalchemy psycopg2-binary pyarrow scikit-learn jupyter
```

Task-specific prerequisites and execution commands are documented in the relevant task README.

## How to Navigate the Repository

Start with this README to understand the overall roadmap and current progress. Then open the README for the task you want to study or execute.

For the current implementation, use the following order:

```text
1. Read Task1/README.md
2. Run Task 1 if the PostgreSQL foundation is not available
3. Read Task2/README.md
4. Run the Task 2 notebooks in numeric order
5. Review the generated reports, charts, metrics, and model artifacts
6. Read `Task3/README.md` and run the inference service after supplying the Task 2 bundle
```

## Contribution and Working Conventions

New tasks should be added as separate directories with their own README. Each task README should explain its purpose, structure, prerequisites, execution order, inputs, outputs, validation checks, and limitations.

Generated files should be kept separate from source code. Notebook names should use numeric prefixes when execution order matters. Commit messages should describe the change clearly, and sensitive or machine-specific files should remain excluded from version control.

## Data Attribution

The Olist dataset should be used and redistributed according to the terms and attribution requirements of its original source. Review the original dataset page and license information before redistributing the raw data or using it outside the intended training context.

## License and Disclaimer

This repository is an educational MLOps training project. The code and workflows are intended for learning, experimentation, and reproducibility practice. They should be reviewed and adapted before being used in a production environment.

The recorded metrics and validation results describe the execution captured in the project. They may change if the source data, filtering rules, feature policy, split strategy, or model configuration changes.
