# Zepto Platform: E-Commerce Data, BI & Support Platform

An end-to-end data and AI engineering platform built for e-commerce catalog operations. The project ingests live product listings, transforms raw currencies and metadata, loads them into an embedded relational database, provides an interactive BI analytics dashboard, and exposes an intelligent catalog retrieval support engine.

---

##  Architecture Overview

1. **Ingestion & ETL Pipeline (`data_pipeline/`)**
   * Multi-page web scraper extracting product titles, categories, pricing, ratings, and stock status.
   * Cleans text encoding artifacts, strips currency symbols via regex, and converts GBP to INR (₹105/GBP).
   * Maps qualitative star ratings (`One` to `Five`) to normalized integer scales (`1` to `5`).
   * Loads structured tabular records directly into an SQLite database (`catalog.db`).

2. **Analytics & BI Dashboard (`analytics/`)**
   * Interactive Streamlit dashboard reading directly from `catalog.db`.
   * Real-time KPI summary cards (Total Inventory, Average INR Price, In-Stock Rate).
   * Category-wise dynamic average price distribution charts.
   * Multi-dimensional filtering by category, maximum price, and minimum star rating.

3. **Customer Support Assistant (`support_assistant/`)**
   * Parameterized catalog search engine querying SQLite inventory using SQL pattern matching.
   * Generates formatted, conversational product recommendations with real-time stock verification.

---

## Tech Stack

* **Language**: Python 3.x
* **Data Processing & Database**: Pandas, SQLite3, Regular Expressions (`re`)
* **Scraping**: Requests, BeautifulSoup4
* **Visualization & UI**: Streamlit, Matplotlib
* **Version Control**: Git, GitHub (Feature Branch Workflow)

---

## Quickstart & Execution

### 1. Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install requests beautifulsoup4 pandas streamlit matplotlib