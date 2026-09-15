import requests
from bs4 import BeautifulSoup
import pandas as pd

# The catalog base URL
BASE_URL = "http://books.toscrape.com/catalogue/"

# Scrape the first 4 pages (20 books/page = 80 books total)
# This satisfies the requirement: >= 60 books across >= 3 categories
PAGES_TO_SCRAPE = 4

book_records = []

print("Starting catalog scraper...")

for page_num in range(1, PAGES_TO_SCRAPE + 1):
    page_url = f"{BASE_URL}page-{page_num}.html"
    print(f"Scraping catalog page {page_num}...")
    
    response = requests.get(page_url)
    if response.status_code != 200:
        print(f"Warning: Could not fetch page {page_num} (status: {response.status_code})")
        continue

    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Each book card lives inside an <article class="product_pod">
    articles = soup.find_all("article", class_="product_pod")
    
    for article in articles:
        # 1. Title: stored inside 'title' attribute of <a> inside <h3>
        title = article.h3.a["title"]
        
        # 2. Price: text inside <p class="price_color"> (e.g., '£51.77')
        price_text = article.find("p", class_="price_color").text.strip()
        
        # 3. Rating: stored as a class name on <p class="star-rating [Rating]">
        rating_classes = article.find("p", class_="star-rating")["class"]
        rating_text = [cls for cls in rating_classes if cls != "star-rating"][0]
        
        # 4. Availability: text inside <p class="instock availability">
        availability_text = article.find("p", class_="instock availability").text.strip()
        
        # 5. Category: fetch the book's detail page to inspect its breadcrumbs
        relative_link = article.h3.a["href"]
        detail_url = BASE_URL + relative_link
        detail_response = requests.get(detail_url)
        detail_soup = BeautifulSoup(detail_response.text, "html.parser")
        
        breadcrumb = detail_soup.find("ul", class_="breadcrumb")
        category = breadcrumb.find_all("li")[2].text.strip() if breadcrumb else "Unknown"
        
        book_records.append({
            "title": title,
            "price_raw": price_text,
            "rating_raw": rating_text,
            "availability_raw": availability_text,
            "category": category
        })

print(f"\nScraping complete! Total books scraped: {len(book_records)}")

# Convert to DataFrame and preview
df_raw = pd.DataFrame(book_records)
print("\nUnique categories found:", df_raw["category"].nunique())
print(df_raw.head())

# Save raw output as offline fallback
df_raw.to_csv("data_pipeline/raw_scraped_books.csv", index=False)
print("Saved raw data to data_pipeline/raw_scraped_books.csv")