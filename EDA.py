import pandas as pd

# Load the dataset
file_path = "/Users/kumarpoudel/Downloads/IMB881.xlsx"
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, "Raw Data-Order and Sample")

# === GENERAL REVIEW ===
# 1. Check dataset shape
print(f"Dataset Shape: {df.shape}")  # (rows, columns)

# 2. Display first 5 rows
print("\nFirst 5 Rows:\n", df.head())

# 3. Check column names and data types
print("\nColumn Info:")
print(df.info())

# 4. Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# 5. Check for duplicate rows
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# 6. Basic statistical summary
print("\nStatistical Summary:\n", df.describe())

# === DATA CLEANING ===
# 1. Remove Duplicate Rows
df = df.drop_duplicates()

# 2. Handle Missing Values in "CustomerOrderNo"
# Handle Missing Values in "CustomerOrderNo"
df = df.assign(CustomerOrderNo=df["CustomerOrderNo"].fillna("Unknown"))
# Replace missing values with "Unknown"
# Alternatively, to drop missing rows:
# df = df.dropna(subset=['CustomerOrderNo'])

# 3. Investigate and Handle Zero Values in "Amount"
zero_amount_rows = df[df["Amount"] == 0]
print(f"\nRows with Amount = 0: {zero_amount_rows.shape[0]}")

# Decide whether to remove rows with zero Amount (optional)
df = df[df["Amount"] > 0]  # Remove rows where Amount is 0

# 4. Verify Data Cleaning
print(f"\nDuplicate Rows After Cleaning: {df.duplicated().sum()}")  # Should be 0
print("\nMissing Values After Cleaning:\n", df.isnull().sum())  # Should be minimal

# Save the cleaned dataset (optional)
cleaned_file_path = "/Users/kumarpoudel/Downloads/cleaned_data.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"\nData cleaning completed! Cleaned file saved as: {cleaned_file_path}")
