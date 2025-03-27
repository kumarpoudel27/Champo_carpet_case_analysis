import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os  # Added import

# Create Plots directory if it doesn't exist
plots_dir = "Plots"
os.makedirs(plots_dir, exist_ok=True)

# Load the dataset
# file_path = "/Users/kumarpoudel/Downloads/IMB881.xlsx" # Changed to relative path
file_path = "IMB881.xlsx"  # Assumes file is in the same directory as the script
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
# Use relative path for cleaned data as well
cleaned_file_path = "cleaned_data.csv"
# cleaned_file_path = "/Users/kumarpoudel/Desktop/Champo_Carpet_Analysis/cleaned_data.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nData cleaning completed! Cleaned file saved as: {cleaned_file_path}")

# === VISUALIZATIONS ===


def plot_pie_chart(column, title, colors=["#ff9999", "#66b3ff", "#99ff99"]):
    """Generic function for pie chart visualization"""
    plt.figure(figsize=(8, 6))
    df[column].value_counts().plot.pie(autopct="%1.1f%%", colors=colors)
    plt.title(title)
    plt.ylabel("")
    # Save plot using a descriptive name based on the title
    save_path = os.path.join(
        plots_dir, f"figure_1_{title.lower().replace(' ', '_')}.png"
    )
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
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
    # Save plot using a descriptive name based on the title
    save_path = os.path.join(plots_dir, f"figure_{title.lower().replace(' ', '_')}.png")
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.show()


# 1. Order Type Distribution
plot_pie_chart("OrderType", "Order Type Distribution")

# 2. Geographical Sales Map
if "Country" in df.columns:
    fig_geo = px.scatter_geo(  # Capture figure object
        df,
        locations="Country",
        locationmode="country names",
        size="Amount",
        title="Geographical Sales Map",
    )
    # Save Plotly figure
    save_path_geo = os.path.join(plots_dir, "figure_2_geographical_sales_map.png")
    try:
        fig_geo.write_image(save_path_geo)
        print(f"Saved plot: {save_path_geo}")
    except Exception as e:
        print(f"Could not save Plotly figure {save_path_geo}. Error: {e}")
        print("Ensure you have 'kaleido' installed (`pip install -U kaleido`)")
    fig_geo.show()


# 3. Product Category Bar Chart
if "ProductCategory" in df.columns:
    plot_bar_chart(
        "ProductCategory", "Product Category Distribution", "Product Category", "Count"
    ) # Saved internally by helper function

# 4. Monthly Sales Trend Line
df["Custorderdate"] = pd.to_datetime(df["Custorderdate"], errors="coerce")
df["MonthYear"] = df["Custorderdate"].dt.to_period("M")
monthly_sales = df.groupby("MonthYear").size()
fig4 = plt.figure(figsize=(12, 6))  # Capture figure object
monthly_sales.plot(marker="o", linestyle="-", color="b")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.grid(True)
# Save plot using figure object
save_path_trend = os.path.join(plots_dir, "figure_4_monthly_sales_trend.png")
fig4.savefig(save_path_trend)
print(f"Saved plot: {save_path_trend}")
plt.show()

# 5. Customer Order Frequency (Top 10)
top_customers = df["CustomerCode"].value_counts().nlargest(10)
plot_bar_chart(
    "CustomerCode",
    "Top Customers by Order Frequency",
    "Customer ID",
    "Number of Orders",
) # Saved internally by helper function

# 6. Price Distribution Box Plot
fig6 = plt.figure(figsize=(10, 6)) # Capture figure object
sns.boxplot(x=df["Amount"])
plt.title("Price Distribution by Product Category")
plt.xlabel("Amount")
# Save plot
save_path_box = os.path.join(plots_dir, "figure_6_price_distribution_boxplot.png")
fig6.savefig(save_path_box)
print(f"Saved plot: {save_path_box}")
plt.show()

# 7. Color Popularity Word Cloud
if "Color" in df.columns:
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(df["Color"].dropna())
    )
    fig7 = plt.figure(figsize=(10, 6)) # Capture figure object
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Color Popularity Word Cloud")
    # Save plot
    save_path_cloud = os.path.join(plots_dir, "figure_7_color_popularity_wordcloud.png")
    fig7.savefig(save_path_cloud)
    print(f"Saved plot: {save_path_cloud}")
    plt.show()

# 8. Order Size Distribution Histogram
fig8 = plt.figure(figsize=(10, 6)) # Capture figure object
sns.histplot(df["QtyRequired"], bins=30, kde=True, color="purple")
plt.title("Order Size Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
# Save plot
save_path_hist = os.path.join(plots_dir, "figure_8_order_size_histogram.png")
fig8.savefig(save_path_hist)
print(f"Saved plot: {save_path_hist}")
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
    fig_elbow = plt.figure(figsize=(10, 6)) # Capture figure object
    plt.plot(range(1, 11), wcss, marker="o", linestyle="-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(range(1, 11))
    plt.grid(True)
    # Save plot
    save_path_elbow = os.path.join(plots_dir, "figure_9_kmeans_elbow_method.png")
    fig_elbow.savefig(save_path_elbow)
    print(f"Saved plot: {save_path_elbow}")
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
    fig_kmeans = plt.figure(figsize=(10, 6)) # Capture figure object
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
    # Save plot
    save_path_kmeans = os.path.join(plots_dir, f"figure_10_customer_segments_k{optimal_k}.png")
    fig_kmeans.savefig(save_path_kmeans)
    print(f"Saved plot: {save_path_kmeans}")
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

# Define functions first (moved from the bottom)
def assign_cluster_names(df, cluster_column, cluster_names):
    """
    Assign meaningful names to clusters.
    Args:
        df (pd.DataFrame): The DataFrame containing the cluster column.
        cluster_column (str): The name of the column with cluster labels.
        cluster_names (dict): A dictionary mapping cluster labels to names.
    Returns:
        pd.DataFrame: The updated DataFrame with a new 'Segment' column.
    """
    if cluster_column in df.columns:
        df["Segment"] = df[cluster_column].map(cluster_names)
    else:
        # Raise error instead of just printing
        raise ValueError(
            f"Error: '{cluster_column}' column is missing. Clustering might have failed."
        )
    return df


def plot_cluster_boxplot(df, cluster_column, value_column, title):
    """
    Plot a boxplot for a given cluster column and value column. Also saves the plot.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        cluster_column (str): The name of the cluster column.
        value_column (str): The name of the value column to plot.
        title (str): The title of the plot.
    """
    if cluster_column in df.columns:
        fig_box_final = plt.figure(figsize=(10, 6)) # Capture figure object
        sns.boxplot(x=cluster_column, y=value_column, data=df)
        plt.title(title)
        # Save plot
        save_path_box_final = os.path.join(plots_dir, f"figure_11_{title.lower().replace(' ', '_')}.png")
        fig_box_final.savefig(save_path_box_final)
        print(f"Saved plot: {save_path_box_final}")
        plt.show()
    else:
        # Raise error instead of just printing
        raise ValueError(
            f"Error: '{cluster_column}' column is missing. Cannot plot boxplot."
        )

# Use the functions
try:
    # Assign cluster names
    df = assign_cluster_names(df, "Cluster", cluster_names)

    # Plot cluster-wise boxplot (now also saves the plot)
    plot_cluster_boxplot(
        df,
        cluster_column="Cluster",
        value_column="Amount",
        title="Order Amount Distribution by Cluster",
    )
except ValueError as e:
    print(e)

# Removed redundant block that was previously here