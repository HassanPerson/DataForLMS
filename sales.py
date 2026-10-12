import os
import pandas as pd
import kagglehub

def load_ecommerce_dataset(dataset_handle):
    """
    Downloads and loads an E-Commerce dataset from Kaggle using kagglehub.
    
    Args:
        dataset_handle (str): The Kaggle dataset path 
    """
    print(f"[*] Downloading e-commerce dataset: {dataset_handle}...")
    
    # Download latest version using kagglehub
    path = kagglehub.dataset_download(dataset_handle)
    print(f"[*] Dataset downloaded to local path: {path}")
    
    # List all files in the downloaded directory
    files = os.listdir(path)
    print(f"[*] Available files in dataset folder: {files}")
    
    # Target the primary transactions file
    target_file = None
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if csv_files:
        # Pick the largest CSV or default to the first match
        target_file = os.path.join(path, csv_files[0])
        print(f"[*] Auto-selected E-Commerce CSV file: {csv_files[0]}")
    else:
        raise FileNotFoundError("No CSV files found in the e-commerce dataset directory.")
            
    # Load into Pandas DataFrame
    print(f"[*] Loading data into Pandas DataFrame...")
    df = pd.read_csv(target_file, low_memory=False)
    
    return df

def perform_ecommerce_eda(df):
    """Performs heavy exploratory data analysis tailored for retail and commerce."""
    print("\n" + "="*60)
    print(" E-COMMERCE DATASET ANALYTICS OVERVIEW ")
    print("="*60)
    
    print(f"\n[+] Total Transactions / Rows: {df.shape[0]:,}")
    print(f"[+] Total Features / Columns: {df.shape[1]}")
    
    print("\n[+] First 5 rows of transactions:")
    display(df.head()) if 'display' in globals() else print(df.head())
    
    print("\n[+] Data Schema & Memory Usage:")
    df.info()
    
    print("\n[+] Statistical Summary (Numerical Metrics like Revenue, Quantity, Discounts):")
    display(df.describe()) if 'display' in globals() else print(df.describe())
    
    print("\n[+] Missing Values Per Feature:")
    missing = df.isnull().sum()
    missing_filtered = missing[missing > 0]
    if not missing_filtered.empty:
        print(missing_filtered)
    else:
        print("No missing values found!")
        
    # Check for common e-commerce columns to provide deeper automated insights
    revenue_col = next((col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower() or 'price' in col.lower()), None)
    category_col = next((col for col in df.columns if 'category' in col.lower() or 'product' in col.lower()), None)
    
    if revenue_col:
        print(f"\n[+] Total Estimated Volume/Revenue ({revenue_col}): ${df[revenue_col].sum():,.2f}" if pd.api.types.is_numeric_dtype(df[revenue_col]) else "")
        
    if category_col:
        print(f"\n[+] Top 5 Product Categories by Frequency:")
        print(df[category_col].value_counts().head(5))

    print("\n" + "="*60)

if __name__ == "__main__":
    # E-Commerce Sales & Customer Analytics Dataset Handle
    DATASET_SLUG = "datascikhan/e-commerce-sales-and-customer-analytics"
    
    try:
        df_ecommerce = load_ecommerce_dataset(DATASET_SLUG)
        perform_ecommerce_eda(df_ecommerce)
    except Exception as e:
        print(f"[!] An error occurred: {e}")
        print("[!] Note: Ensure your local environment has active Kaggle API credentials (~/.kaggle/kaggle.json).")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate Synthetic Sales Data
np.random.seed(42)
n_rows = 1000

dates = pd.date_range(start="2025-01-01", end="2025-12-31", periods=n_rows)
categories = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Sports"]
regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]

df = pd.DataFrame({
    "Date": dates,
    "Category": np.random.choice(categories, size=n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    "Region": np.random.choice(regions, size=n_rows, p=[0.4, 0.3, 0.2, 0.1]),
    "Units_Sold": np.random.randint(1, 15, size=n_rows),
    "Unit_Price": np.random.uniform(10.0, 300.0, size=n_rows).round(2)
})

# Calculate total revenue per transaction
df["Revenue"] = df["Units_Sold"] * df["Unit_Price"]
df["Month"] = df["Date"].dt.to_period("M")

# 2. Set Up Plot Style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Randomized Retail Sales Analysis Dashboard", fontsize=16, fontweight="bold")

# Graph 1: Monthly Total Revenue Trend
monthly_sales = df.groupby("Month")["Revenue"].sum().reset_index()
monthly_sales["Month"] = monthly_sales["Month"].astype(str)

sns.lineplot(
    data=monthly_sales, 
    x="Month", 
    y="Revenue", 
    marker="o", 
    color="#1f77b4", 
    linewidth=2.5, 
    ax=axes[0, 0]
)
axes[0, 0].set_title("Monthly Revenue Trend", fontweight="bold")
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("Revenue ($)")
axes[0, 0].tick_params(axis="x", rotation=45)

# Graph 2: Total Revenue by Product Category
category_sales = df.groupby("Category")["Revenue"].sum().reset_index().sort_values(by="Revenue", ascending=False)

sns.barplot(
    data=category_sales, 
    x="Revenue", 
    y="Category", 
    palette="viridis", 
    ax=axes[0, 1]
)
axes[0, 1].set_title("Total Revenue by Category", fontweight="bold")
axes[0, 1].set_xlabel("Revenue ($)")
axes[0, 1].set_ylabel("Category")

# Graph 3: Regional Sales Distribution
region_sales = df.groupby("Region")["Revenue"].sum().reset_index()

axes[1, 0].pie(
    region_sales["Revenue"], 
    labels=region_sales["Region"], 
    autopct="%1.1f%%", 
    startangle=140, 
    colors=sns.color_palette("pastel")
)
axes[1, 0].set_title("Revenue Share by Region", fontweight="bold")

# Graph 4: Units Sold Distribution by Category (Boxplot)
sns.boxplot(
    data=df, 
    x="Category", 
    y="Units_Sold", 
    palette="Set2", 
    ax=axes[1, 1]
)
axes[1, 1].set_title("Units Sold Distribution per Order", fontweight="bold")
axes[1, 1].set_xlabel("Category")
axes[1, 1].set_ylabel("Units Sold")
axes[1, 1].tick_params(axis="x", rotation=30)

# 3. Display Plots
plt.tight_layout()
plt.show()