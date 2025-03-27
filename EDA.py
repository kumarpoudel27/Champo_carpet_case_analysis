import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

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
df["CustomerOrderNo"] = df["CustomerOrderNo"].fillna("Unknown")

# 3. Investigate and Handle Zero Values in "Amount"
zero_amount_rows = df[df["Amount"] == 0]
print(f"\nRows with Amount = 0: {zero_amount_rows.shape[0]}")

# Remove rows where Amount is 0
df = df[df["Amount"] > 0]

# 4. Handle Missing 'Country' and 'ProductCategory' Columns
if "Country" not in df.columns:
    print(
        "The 'Country' column is missing in the dataset. Skipping geographical map visualization."
    )
else:
    # If 'Country' exists, handle missing values
    df["Country"] = df["Country"].fillna("Unknown")

if "ProductCategory" not in df.columns:
    print(
        "The 'ProductCategory' column is missing in the dataset. Skipping product category bar chart."
    )
else:
    # If 'ProductCategory' exists, handle missing values
    df["ProductCategory"] = df["ProductCategory"].fillna("Unknown")

# 5. Verify Data Cleaning
print(f"\nDuplicate Rows After Cleaning: {df.duplicated().sum()}")  # Should be 0
print("\nMissing Values After Cleaning:\n", df.isnull().sum())  # Should be minimal

# Save the cleaned dataset (optional)
cleaned_file_path = "/Users/kumarpoudel/Downloads/cleaned_data.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nData cleaning completed! Cleaned file saved as: {cleaned_file_path}")

# === EDA ANALYSIS ===

# 1. Data Overview
print("\nData Overview:")
print(df.info())

# 2. Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# 3. Summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# 4. Check unique values for categorical columns
print("\nUnique Values in Categorical Columns:")
print(df.select_dtypes(include=["object"]).nunique())

# === VISUALIZATIONS ===

# 1. Order Type Distribution Pie Chart
plt.figure(figsize=(8, 6))
df["OrderType"].value_counts().plot.pie(
    autopct="%1.1f%%", colors=["#ff9999", "#66b3ff", "#99ff99"]
)
plt.title("Order Type Distribution")
plt.ylabel("")
plt.show()

# 2. Geographical Sales Map
if "Country" in df.columns:
    fig = px.scatter_geo(
        df,
        locations="Country",
        locationmode="country names",
        size="Amount",
        title="Geographical Sales Map",
    )
    fig.show()

# 3. Product Category Bar Chart
if "ProductCategory" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=df["ProductCategory"].value_counts().index,
        y=df["ProductCategory"].value_counts().values,
        palette="viridis",
    )
    plt.title("Product Category Distribution")
    plt.xlabel("Product Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# 4. Monthly Sales Trend Line
df["Custorderdate"] = pd.to_datetime(df["Custorderdate"], errors="coerce")
df["MonthYear"] = df["Custorderdate"].dt.to_period("M")
monthly_sales = df.groupby("MonthYear").size()
plt.figure(figsize=(12, 6))
monthly_sales.plot(marker="o", linestyle="-")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.show()

# 5. Customer Order Frequency Bar Chart
top_customers = df["CustomerCode"].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_customers.index,
    y=top_customers.values,
    hue=top_customers.index,
    palette="coolwarm",
    legend=False,
)
plt.title("Top Customers by Order Frequency")
plt.xlabel("Customer ID")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.show()

# 6. Price Distribution Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Amount"])
plt.title("Price Distribution by Product Category")
plt.xlabel("Amount")
plt.show()

# 7. Color Popularity Word Cloud
if "Color" in df.columns:
    text = " ".join(df["Color"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Color Popularity Word Cloud")
    plt.show()

# 8. Order Size Distribution Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df["QtyRequired"], bins=30, kde=True, color="purple")
plt.title("Order Size Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()
