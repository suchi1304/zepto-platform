import sqlite3

DB_PATH = "data_pipeline/catalog.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name like row['title']
    return conn

def search_books(keyword=None, max_price=None, min_rating=None, limit=5):
    """Fetch matching books directly from catalog.db based on customer intent."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT title, category, price_inr, star_rating_num, in_stock FROM books WHERE in_stock = 1"
    params = []

    if keyword:
        query += " AND (title LIKE ? OR category LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%"])

    if max_price:
        query += " AND price_inr <= ?"
        params.append(max_price)

    if min_rating:
        query += " AND star_rating_num >= ?"
        params.append(min_rating)

    query += " ORDER BY star_rating_num DESC, price_inr ASC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]

def support_agent(user_message, keyword=None, max_price=None, min_rating=None):
    """Simulates support bot combining catalog search with customer-facing reply."""
    matches = search_books(keyword=keyword, max_price=max_price, min_rating=min_rating)

    if not matches:
        return f"Hi! I searched our catalog for '{keyword or 'books'}', but couldn't find matches under Rs. {max_price or 'any'}. Would you like to check other categories?"

    response = f"Hello! Based on your request, here are top recommendations from our Zepto catalog:\n"
    for idx, b in enumerate(matches, 1):
        stars = "★" * b["star_rating_num"]
        response += f"{idx}. {b['title']} ({b['category']}) - Rs. {b['price_inr']:,.2f} | Rating: {stars} ({b['star_rating_num']}/5)\n"

    response += "\nAll these items are currently in stock for immediate delivery!"
    return response

if __name__ == "__main__":
    print("--- Zepto Support Assistant Test Run ---")
    
    # Test Case 1: Search for Poetry books under Rs. 5500
    print("\nCustomer: Looking for poetry books under Rs. 5500")
    reply = support_agent("Looking for poetry books", keyword="Poetry", max_price=5500.0)
    print(reply)

    print("\n" + "="*50)

    # Test Case 2: Search for 5-star rated books
    print("\nCustomer: Show me 5-star rated books")
    reply = support_agent("Show me top rated books", min_rating=5)
    print(reply)