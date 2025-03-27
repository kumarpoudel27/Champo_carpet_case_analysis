import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
file_path = "/Users/kumarpoudel/Downloads/IMB881.xlsx"
df = pd.read_excel(file_path, sheet_name="Raw Data-Order and Sample")

# === GENERAL REVIEW ===
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 Rows:\n", df.head())
print("\nColumn Info:")
df.info()
print("\nMissing Values:\n", df.isnull().sum())
print(f"\nDuplicate Rows: {df.duplicated().sum()}")
print("\nStatistical Summary:\n", df.describe())

# === DATA CLEANING ===
df.drop_duplicates(inplace=True)
df.fillna(
    {"CustomerOrderNo": "Unknown", "Country": "Unknown", "ProductCategory": "Unknown"},
    inplace=True,
)

print(f"\nDuplicate Rows After Cleaning: {df.duplicated().sum()}")
print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Save cleaned dataset
cleaned_file_path = "/Users/kumarpoudel/Desktop/Champo_Carpet_Analysis/cleaned_data.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nData cleaning completed! Cleaned file saved as: {cleaned_file_path}")

# === VISUALIZATIONS ===


def plot_pie_chart(column, title, colors=["#ff9999", "#66b3ff", "#99ff99"]):
    """Generic function for pie chart visualization"""
    plt.figure(figsize=(8, 6))
    df[column].value_counts().plot.pie(autopct="%1.1f%%", colors=colors)
    plt.title(title)
    plt.ylabel("")
    plt.show()


def plot_bar_chart(column, title, xlabel, ylabel):
    """Generic function for bar chart visualization"""
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=df[column].value_counts().index,
        y=df[column].value_counts().values,
        palette="viridis",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()


# 1. Order Type Distribution
plot_pie_chart("OrderType", "Order Type Distribution")

# 2. Geographical Sales Map
if "Country" in df.columns:
    px.scatter_geo(
        df,
        locations="Country",
        locationmode="country names",
        size="Amount",
        title="Geographical Sales Map",
    ).show()

# 3. Product Category Bar Chart
if "ProductCategory" in df.columns:
    plot_bar_chart(
        "ProductCategory", "Product Category Distribution", "Product Category", "Count"
    )

# 4. Monthly Sales Trend Line
df["Custorderdate"] = pd.to_datetime(df["Custorderdate"], errors="coerce")
df["MonthYear"] = df["Custorderdate"].dt.to_period("M")
monthly_sales = df.groupby("MonthYear").size()
plt.figure(figsize=(12, 6))
monthly_sales.plot(marker="o", linestyle="-", color="b")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 5. Customer Order Frequency (Top 10)
top_customers = df["CustomerCode"].value_counts().nlargest(10)
plot_bar_chart(
    "CustomerCode",
    "Top Customers by Order Frequency",
    "Customer ID",
    "Number of Orders",
)

# 6. Price Distribution Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Amount"])
plt.title("Price Distribution by Product Category")
plt.xlabel("Amount")
plt.show()

# 7. Color Popularity Word Cloud
if "Color" in df.columns:
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(df["Color"].dropna())
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

# === K-MEANS CLUSTERING ===


def perform_kmeans_clustering(data, features, k_values=[3, 4]):
    """Function to perform k-means clustering and visualize results"""
    df_cluster = data[features].dropna().copy()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)

    # Compute Within-Cluster Sum of Squares (WCSS) for different k values
    wcss = [
        KMeans(n_clusters=k, random_state=42, n_init=10).fit(df_scaled).inertia_
        for k in range(1, 11)
    ]

    # Elbow Method Chart
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker="o", linestyle="-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()

    # Run clustering for selected k values
    optimal_k = k_values[0]  # Choose the first value from k_values
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)

    # Assign cluster labels to original DataFrame
    df_cluster["Cluster"] = cluster_labels
    data = data.merge(
        df_cluster[["Cluster"]], left_index=True, right_index=True, how="left"
    )

    # Visualize clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df_cluster["Amount"],
        y=df_cluster["QtyRequired"],
        hue=df_cluster["Cluster"],
        palette="viridis",
        s=100,
    )
    plt.title(f"Customer Segments (k={optimal_k})")
    plt.xlabel("Order Amount ($)")
    plt.ylabel("Quantity Required")
    plt.grid(True)
    plt.show()

    # Cluster statistics
    print(
        f"\nCluster Statistics for k={optimal_k}:\n",
        df_cluster.groupby("Cluster")[features].agg(["mean", "median", "count"]),
    )

    # Compute silhouette score
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    print(f"Silhouette Score for k={optimal_k}: {silhouette_avg:.3f}")

    return data  # Return updated dataframe with clusters


# Run K-means Clustering and update df
df = perform_kmeans_clustering(df, ["Amount", "QtyRequired"])

# === BUSINESS INTERPRETATION ===
print("\n" + "=" * 50)
print("BUSINESS INTERPRETATION SUGGESTIONS")
print("=" * 50)
print(
    """
1. High-Value, Low-Quantity: Customers who order expensive items in small quantities.
2. Medium-Value, Medium-Quantity: Regular customers with moderate orders.
3. Low-Value, High-Quantity: Bulk buyers of cheaper items.
4. (For k=4) Potential sub-segment: Could represent special cases or outliers.
"""
)

# Assign meaningful cluster names
cluster_names = {
    0: "High-Value, Low-Quantity",
    1: "Medium-Value, Medium-Quantity",
    2: "Low-Value, High-Quantity",
}

if "Cluster" in df.columns:
    df["Segment"] = df["Cluster"].map(cluster_names)
else:
    print("Error: 'Cluster' column is missing. Clustering might have failed.")

# Cluster-wise boxplot
if "Cluster" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Cluster", y="Amount", data=df)
    plt.title("Order Amount Distribution by Cluster")
    plt.show()
