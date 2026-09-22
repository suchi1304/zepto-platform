import os
import sqlite3
import pandas as pd

DB_PATH = "data_pipeline/catalog.db"
RAW_CSV = "data_pipeline/raw_scraped_books.csv"
CLEAN_CSV = "data_pipeline/cleaned_books.csv"

# 1. Load raw data
df_raw = pd.read_csv(RAW_CSV)

# Normalize column names to lowercase
df_raw.columns = [c.lower().strip() for c in df_raw.columns]

# Detect price column dynamically
price_col = None
for col in ["price", "price_gbp", "book_price"]:
    if col in df_raw.columns:
        price_col = col
        break

# Detect rating column dynamically
rating_col = None
for col in ["rating", "rating_stars", "star_rating"]:
    if col in df_raw.columns:
        rating_col = col
        break

# Detect availability column dynamically
avail_col = None
for col in ["availability", "in_stock", "stock"]:
    if col in df_raw.columns:
        avail_col = col
        break

# Detect category column dynamically
cat_col = None
for col in ["category", "category_name"]:
    if col in df_raw.columns:
        cat_col = col
        break

df_clean = df_raw.copy()

# Price cleaning
if price_col:
    df_clean["price_gbp"] = (
        df_clean[price_col]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )
else:
    df_clean["price_gbp"] = 25.0

# Rating mapping
rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
if rating_col:
    df_clean["rating_stars"] = df_clean[rating_col].astype(str).map(rating_map).fillna(3).astype(int)
else:
    df_clean["rating_stars"] = 3

# In stock cleaning
if avail_col:
    df_clean["in_stock"] = df_clean[avail_col].astype(str).apply(lambda x: 1 if "in stock" in x.lower() or x.strip() == "1" else 0)
else:
    df_clean["in_stock"] = 1

# Category cleaning
if cat_col:
    df_clean["category"] = df_clean[cat_col].fillna("General").astype(str).str.strip()
else:
    df_clean["category"] = "General"

# Title cleaning
title_col = "title" if "title" in df_raw.columns else df_raw.columns[0]
df_clean["title"] = df_clean[title_col].astype(str).str.strip()

# Save cleaned CSV
df_clean[["title", "price_gbp", "rating_stars", "in_stock", "category"]].to_csv(CLEAN_CSV, index=False)
print("Cleaned dataset saved to", CLEAN_CSV)

# 2. SQLite 2-Table Normalized Schema
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.executescript("""
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    rating_stars INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (category_id)
);
""")

categories = sorted(df_clean["category"].unique())
category_to_id = {}
for cat in categories:
    cursor.execute("INSERT INTO categories (category_name) VALUES (?)", (cat,))
    category_to_id[cat] = cursor.lastrowid

books_data = []
for _, row in df_clean.iterrows():
    books_data.append((
        row["title"],
        float(row["price_gbp"]),
        int(row["rating_stars"]),
        int(row["in_stock"]),
        category_to_id[row["category"]]
    ))

cursor.executemany("""
    INSERT INTO books (title, price_gbp, rating_stars, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?)
""", books_data)

conn.commit()
print(f"Loaded {len(categories)} categories and {len(books_data)} books into {DB_PATH}")

# 3. 5 Analytical SQL Queries
queries = {
    "Query 1: Top 5 Most Expensive Books": """
        SELECT b.title, c.category_name, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.price_gbp DESC
        LIMIT 5;
    """,
    "Query 2: Average Price Per Star Rating": """
        SELECT rating_stars, ROUND(AVG(price_gbp), 2) AS avg_price, COUNT(*) AS count
        FROM books
        GROUP BY rating_stars
        ORDER BY rating_stars DESC;
    """,
    "Query 3: Total Books and Average Price by Category": """
        SELECT c.category_name, COUNT(b.book_id) AS total_books, ROUND(AVG(b.price_gbp), 2) AS avg_price
        FROM categories c
        LEFT JOIN books b ON c.category_id = b.category_id
        GROUP BY c.category_name
        ORDER BY total_books DESC;
    """,
    "Query 4: High-Value Categories (>£30 Avg Price)": """
        SELECT c.category_name, ROUND(AVG(b.price_gbp), 2) AS avg_price, COUNT(b.book_id) AS total_books
        FROM categories c
        JOIN books b ON c.category_id = b.category_id
        GROUP BY c.category_name
        HAVING AVG(b.price_gbp) > 30.0
        ORDER BY avg_price DESC;
    """,
    "Query 5: Cheapest 5-Star Book In Stock": """
        SELECT b.title, c.category_name, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating_stars = 5 AND b.in_stock = 1
        ORDER BY b.price_gbp ASC
        LIMIT 1;
    """
}

for title, q in queries.items():
    print(f"\n--- {title} ---")
    res = pd.read_sql_query(q, conn)
    print(res.to_string(index=False))

# 4. Check Equivalence (SQL JOIN vs Pandas merge)
sql_joined = pd.read_sql_query("""
    SELECT b.title, b.price_gbp, b.rating_stars, c.category_name
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    ORDER BY b.title;
""", conn)

df_books_raw = pd.read_sql_query("SELECT * FROM books", conn)
df_cats_raw = pd.read_sql_query("SELECT * FROM categories", conn)

pandas_merged = pd.merge(
    df_books_raw, df_cats_raw, on="category_id"
)[["title", "price_gbp", "rating_stars", "category_name"]].sort_values("title").reset_index(drop=True)

print(f"\n=== EQUIVALENCE CHECK: SQL JOIN == pd.merge -> {sql_joined.equals(pandas_merged)} ===")
conn.close()