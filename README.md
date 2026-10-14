# 🛍️ Mall Customer Segmentation

A web app that groups mall customers into 4 segments using **K-Means clustering**, built with Python, scikit-learn and Streamlit.

🔗 **Live app:** https://mall-customer-datasets-zvvfvpbkl3ntyp9nbkmhss.streamlit.app/

## Features
- 📊 **Dashboard:** customer segments scatter plot, segment share, average profile and gender mix
- 🎯 **Predict:** enter income and spending score to find a customer's segment, with a match % for every segment
- 📁 **Batch:** upload a CSV and segment many customers at once
- 🧩 **Segments:** profile card and radar chart for each segment

## Segments

| Segment | Avg Income (k$) | Avg Spending Score | Customers |
|---|---|---|---|
| Wealthy Savers | 105.5 | 6.3 | 149 |
| Affluent Regulars | 78.1 | 33.2 | 218 |
| Average Shoppers | 44.8 | 47.5 | 354 |
| Trendy Spenders | 30.5 | 63.3 | 279 |

## How it works
1. Missing values are filled (median for numbers, most common value for text).
2. **Annual Income** and **Spending Score** are scaled with `StandardScaler`.
3. `KMeans` with 4 clusters groups the customers.
4. Each cluster is named from its centre (income and spending level).

## Tech stack
Python, Pandas, NumPy, scikit-learn, Plotly, Streamlit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – Streamlit app (trains the model and shows the dashboard)
- `store_customers.csv` – customer dataset
- `requirements.txt` – Python packages
