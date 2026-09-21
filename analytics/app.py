import streamlit as st
import sqlite3
import pandas as pd

# Page setup
st.set_page_config(page_title="Zepto Catalog Analytics", layout="wide")
st.title("Zepto Books Catalog Dashboard")

# Pull clean data from database
conn = sqlite3.connect("data_pipeline/catalog.db")
df = pd.read_sql("SELECT * FROM books", conn)
conn.close()

# Sidebar controls
st.sidebar.header("Filter Catalog")

# Category dropdown
category_list = ["All"] + sorted(df["category"].unique().tolist())
selected_cat = st.sidebar.selectbox("Category", category_list)

# Price slider
max_p = float(df["price_inr"].max())
min_p = float(df["price_inr"].min())
price_cutoff = st.sidebar.slider("Max Price (INR)", min_p, max_p, max_p, step=50.0)

# Rating slider
rating_cutoff = st.sidebar.slider("Minimum Rating", 1, 5, 1)

# Apply user filters
filtered = df[(df["price_inr"] <= price_cutoff) & (df["star_rating_num"] >= rating_cutoff)]

if selected_cat != "All":
    filtered = filtered[filtered["category"] == selected_cat]

# Summary KPI cards
c1, c2, c3 = st.columns(3)
c1.metric("Total Items", len(filtered))

avg_price = filtered["price_inr"].mean() if len(filtered) > 0 else 0
c2.metric("Avg Price", f"Rs. {avg_price:,.2f}")

stock_rate = (filtered["in_stock"].mean() * 100) if len(filtered) > 0 else 0
c3.metric("In-Stock %", f"{stock_rate:.1f}%")

st.divider()

# Price comparison across categories
st.subheader("Average Price by Category (INR)")
if not filtered.empty:
    chart_data = filtered.groupby("category")["price_inr"].mean().round(2)
    st.bar_chart(chart_data)
else:
    st.info("No books match the current filters.")

# Data table view
st.subheader("Catalog Items")
st.dataframe(filtered, use_container_width=True)