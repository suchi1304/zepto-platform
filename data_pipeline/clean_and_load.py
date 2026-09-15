import sqlite3
import pandas as pd

# 1. Read the raw scraped CSV with UTF-8 encoding
raw_path = "data_pipeline/raw_scraped_books.csv"
print(f"Reading raw data from {raw_path}...")
df = pd.read_csv(raw_path, encoding="utf-8")

# 2. Clean the Price column using Regular Expressions
# r"([\d\.]+)" extracts only digits and decimals, discarding symbols like '£' or 'Â'
df["price_gbp"] = (
    df["price_raw"]
    .str.extract(r"([\d\.]+)", expand=False)
    .astype(float)
)

# Convert GBP to INR (1 GBP ≈ ₹105.00)
GBP_TO_INR_RATE = 105.00
df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

# 3. Map Star Ratings to Numerical Integers (1 to 5)
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}
df["star_rating_num"] = df["rating_raw"].map(rating_map).fillna(0).astype(int)

# 4. Clean Availability Status to Binary (1 = In Stock, 0 = Out of Stock)
df["in_stock"] = df["availability_raw"].str.contains("In stock", case=False).astype(int)

# 5. Clean Title and Category strings
df["title"] = df["title"].str.strip()
df["category"] = df["category"].str.strip()

# Select final structured columns
cleaned_df = df[[
    "title",
    "category",
    "price_gbp",
    "price_inr",
    "star_rating_num",
    "in_stock"
]].copy()

print("\nCleaned Data Preview:")
print(cleaned_df.head())
print(f"\nTotal rows cleaned: {len(cleaned_df)}")

# 6. Save directly into SQLite Database
db_path = "data_pipeline/catalog.db"
conn = sqlite3.connect(db_path)
cleaned_df.to_sql("books", conn, if_exists="replace", index=False)
conn.close()

print(f"\nSuccessfully created and loaded table 'books' into {db_path}!")

# Also save a cleaned CSV
cleaned_df.to_csv("data_pipeline/cleaned_books.csv", index=False, encoding="utf-8")
print("Saved cleaned CSV to data_pipeline/cleaned_books.csv")