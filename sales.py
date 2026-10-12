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