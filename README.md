# Zepto Platform Capstone Project

A comprehensive, production-ready three-module system comprising an automated data engineering pipeline, a predictive machine learning analytics suite, and a grounded GenAI support assistant.

---

## Architecture Overview

```text
zepto-platform/
├── data_pipeline/             # Module 1: Web Scraper & SQL Storage (25 Marks)
│   ├── scraper.py             # Books to Scrape catalog crawler
│   ├── clean_and_load.py      # Normalization, SQLite ETL & 5 SQL analytical queries
│   ├── catalog.db             # 2-Table SQLite relational DB (categories, books)
│   ├── cleaned_books.csv      # Cleaned flat tabular export
│   └── README.md              # ER schema & SQL vs Pandas documentation
│
├── analytics/                 # Module 2: Titanic Predictive Suite & Dashboard (50 Marks)
│   ├── run_eda.py             # EDA, IQR outliers, skewness, 4 multivariate plots
│   ├── run_modeling.py        # Leak-free ColumnTransformer, 3 classifiers, RF tuning, fare regression
│   ├── titanic_pipeline.joblib# End-to-end fitted deployment artifact
│   ├── titanic.csv            # Committed offline fallback dataset
│   ├── charts/                # Exported figures (plots, decision tree, residuals)
│   └── README.md              # Missing-value strategy, metric tables & recommendations
│
└── support_assistant/         # Module 3: GenAI RAG Microservice (25 Marks)
    ├── docs/                  # 8 domain-specific operational Zepto policy documents
    ├── setup_docs.py          # Deterministic policy document generator
    ├── ingest.py              # ChromaDB vector store with all-MiniLM-L6-v2 embeddings
    ├── main.py                # LangGraph 3-node StateGraph router & FastAPI application
    ├── Dockerfile             # Containerized microservice definition
    ├── requirements.txt       # Production dependencies
    └── README.md              # LangGraph architecture & live request/response transcripts
    