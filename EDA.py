import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
# import traceback # Removed traceback import

# Create Plots directory if it doesn't exist
plots_dir = "Plots"
os.makedirs(plots_dir, exist_ok=True)

# Load the dataset
file_path = "IMB881.xlsx"
try:
    df = pd.read_excel(file_path, sheet_name="Raw Data-Order and Sample")
    print("Successfully loaded data.")
except FileNotFoundError:
    print(f"ERROR: Input file not found at {file_path}. Please ensure it's in the correct directory.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to load data from {file_path}. Error: {e}")
    exit()


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
# Assuming fillna targets are correct based on raw data structure
df.fillna(
    {"CustomerOrderNo": "Unknown", "Country": "Unknown", "ProductCategory": "Unknown"},
    inplace=True,
)

print(f"\nDuplicate Rows After Cleaning: {df.duplicated().sum()}")
print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Save cleaned dataset
cleaned_file_path = "cleaned_data.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nData cleaning completed! Cleaned file saved as: {cleaned_file_path}")

# === VISUALIZATIONS ===
print("\n--- Starting Visualizations ---")

def plot_pie_chart(column, title, colors=["#ff9999", "#66b3ff", "#99ff99"]):
    """Generic function for pie chart visualization"""
    print(f"\nGenerating Plot 1: {title}")
    try:
        plt.figure(figsize=(8, 6))
        value_counts = df[column].value_counts()
        if value_counts.empty:
            print(f"WARNING: No data found for column '{column}' to generate pie chart.")
            plt.close()
            return
        value_counts.plot.pie(autopct="%1.1f%%", colors=colors)
        plt.title(title)
        plt.ylabel("")
        save_path = os.path.join(
            plots_dir, f"figure_1_{title.lower().replace(' ', '_')}.png"
        )
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.show()
    except Exception as e:
        print(f"ERROR generating/saving Plot 1 ({title}): {e}")
        # Removed traceback


def plot_bar_chart(column, title, xlabel, ylabel):
    """Generic function for bar chart visualization"""
    print(f"\nGenerating Bar Chart: {title}")
    try:
        plt.figure(figsize=(10, 6))
        value_counts = df[column].value_counts()
        if value_counts.empty:
            print(f"WARNING: No data found for column '{column}' to generate bar chart.")
            plt.close()
            return
        sns.barplot(
            x=value_counts.index,
            y=value_counts.values,
            palette="viridis",
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        filename = f"figure_{column.lower()}_{title.lower().replace(' ', '_')}.png"
        save_path = os.path.join(plots_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.show()
    except Exception as e:
        print(f"ERROR generating/saving Bar Chart ({title}): {e}")
        # Removed traceback


# 1. Order Type Distribution
plot_pie_chart("OrderType", "Order Type Distribution")

# 2. Geographical Sales Map
print("\nGenerating Plot 2: Geographical Sales Map")
if "CountryName" in df.columns and "Amount" in df.columns:
    geo_data = df[['CountryName', 'Amount']].dropna()
    geo_data = geo_data[geo_data['Amount'] > 0]

    if not geo_data.empty:
        try:
            fig_geo = px.scatter_geo(
                geo_data,
                locations="CountryName",
                locationmode="country names",
                size="Amount",
                title="Geographical Sales Map",
            )
            save_path_geo = os.path.join(plots_dir, "figure_2_geographical_sales_map.png")
            try:
                fig_geo.write_image(save_path_geo)
                print(f"Saved plot: {save_path_geo}")
            except Exception as e_save:
                # Simplified error message
                print(f"ERROR saving Plotly figure {save_path_geo}. Error: {e_save}")
                print("Ensure 'kaleido' is installed and functional (`pip install -U kaleido`)")
            fig_geo.show()
        except Exception as e_plot:
            print(f"ERROR generating Plotly figure for Geo Map: {e_plot}")
            # Removed traceback
    else:
        print("WARNING: No valid data (CountryName, Amount > 0) found for Geographical Sales Map.")
else:
    print("WARNING: Skipping Geographical Sales Map - 'CountryName' or 'Amount' column missing.")


# 3. Product Category Bar Chart
print("\nGenerating Plot 3: Product Category Distribution")
if "ProductCategory" in df.columns:
    try:
        plot_bar_chart(
            "ProductCategory", "Product Category Distribution", "Product Category", "Count"
        )
    except Exception as e:
        print(f"ERROR during Plot 3 generation/saving: {e}")
        # Removed traceback
else:
    # This warning might still appear if the check fails for unknown reasons
    print("WARNING: Skipping Product Category Bar Chart - 'ProductCategory' column missing check failed unexpectedly.")


# 4. Monthly Sales Trend Line
print("\nGenerating Plot 4: Monthly Sales Trend")
try:
    df["Custorderdate"] = pd.to_datetime(df["Custorderdate"], errors="coerce")
    df["MonthYear"] = df["Custorderdate"].dt.to_period("M")
    monthly_sales = df.groupby("MonthYear").size()
    if monthly_sales.empty:
        print("WARNING: No monthly sales data found for trend line.")
    else:
        fig4 = plt.figure(figsize=(12, 6))
        monthly_sales.plot(marker="o", linestyle="-", color="b")
        plt.title("Monthly Sales Trend")
        plt.xlabel("Month")
        plt.ylabel("Number of Orders")
        plt.xticks(rotation=45)
        plt.grid(True)
        save_path_trend = os.path.join(plots_dir, "figure_4_monthly_sales_trend.png")
        fig4.savefig(save_path_trend)
        print(f"Saved plot: {save_path_trend}")
        plt.show()
except Exception as e:
    print(f"ERROR generating/saving Plot 4: {e}")
    # Removed traceback


# 5. Customer Order Frequency (Top 10)
print("\nGenerating Plot 5: Top Customers by Order Frequency")
if "CustomerCode" in df.columns:
    try:
        plot_bar_chart(
            "CustomerCode",
            "Top Customers by Order Frequency",
            "Customer ID",
            "Number of Orders",
        )
    except Exception as e:
        print(f"ERROR during Plot 5 generation/saving: {e}")
        # Removed traceback
else:
    print("WARNING: Skipping Top Customers plot - 'CustomerCode' column missing.")


# 6. Price Distribution Box Plot
print("\nGenerating Plot 6: Price Distribution Box Plot")
if "Amount" in df.columns:
    try:
        fig6 = plt.figure(figsize=(10, 6))
        sns.boxplot(x=df["Amount"])
        plt.title("Price Distribution")
        plt.xlabel("Amount")
        save_path_box = os.path.join(plots_dir, "figure_6_price_distribution_boxplot.png")
        fig6.savefig(save_path_box)
        print(f"Saved plot: {save_path_box}")
        plt.show()
    except Exception as e:
        print(f"ERROR generating/saving Plot 6: {e}")
        # Removed traceback
else:
    print("WARNING: Skipping Price Distribution plot - 'Amount' column missing.")


# 7. Color Popularity Word Cloud
print("\nGenerating Plot 7: Color Popularity Word Cloud")
if "ColorName" in df.columns:
    try:
        color_text = df["ColorName"].dropna()
        if not color_text.empty:
            text_to_generate = " ".join(color_text)
            if text_to_generate.strip():
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_to_generate)
                fig7 = plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title("Color Popularity Word Cloud")
                save_path_cloud = os.path.join(plots_dir, "figure_7_color_popularity_wordcloud.png")
                fig7.savefig(save_path_cloud)
                print(f"Saved plot: {save_path_cloud}")
                plt.show()
            else:
                 print("WARNING: Color data is present but results in empty or whitespace-only string after join/strip. Skipping word cloud.")
        else:
            print("WARNING: No non-null color data found for word cloud.")
    except Exception as e:
        print(f"ERROR generating/saving Plot 7: {e}")
        # Removed traceback
else:
    print("WARNING: Skipping Color Word Cloud - 'ColorName' column missing.")


# 8. Order Size Distribution Histogram
print("\nGenerating Plot 8: Order Size Distribution Histogram")
if "QtyRequired" in df.columns:
    try:
        fig8 = plt.figure(figsize=(10, 6))
        sns.histplot(df["QtyRequired"], bins=30, kde=True, color="purple")
        plt.title("Order Size Distribution")
        plt.xlabel("Quantity")
        plt.ylabel("Frequency")
        save_path_hist = os.path.join(plots_dir, "figure_8_order_size_histogram.png")
        fig8.savefig(save_path_hist)
        print(f"Saved plot: {save_path_hist}")
        plt.show()
    except Exception as e:
        print(f"ERROR generating/saving Plot 8: {e}")
        # Removed traceback
else:
    print("WARNING: Skipping Order Size Histogram - 'QtyRequired' column missing.")


# === K-MEANS CLUSTERING ===
print("\n--- Starting K-Means Clustering ---")

def perform_kmeans_clustering(data, features, k_values=[3, 4]):
    """Function to perform k-means clustering and visualize results"""
    print(f"\nPerforming K-Means with features: {features}")
    try:
        df_cluster = data[features].dropna().copy()
        if df_cluster.empty or len(df_cluster) < min(k_values):
             print(f"WARNING: Not enough valid data points ({len(df_cluster)}) for clustering with features {features}. Skipping.")
             return data

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_cluster)

        print("Calculating WCSS for Elbow Method...")
        wcss = [
            KMeans(n_clusters=k, random_state=42, n_init=10).fit(df_scaled).inertia_
            for k in range(1, min(11, len(df_cluster)))
        ]

        print("Generating Plot 9: Elbow Method")
        fig_elbow = plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(wcss) + 1), wcss, marker="o", linestyle="-")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
        plt.title("Elbow Method for Optimal k")
        plt.xticks(range(1, len(wcss) + 1))
        plt.grid(True)
        save_path_elbow = os.path.join(plots_dir, "figure_9_kmeans_elbow_method.png")
        fig_elbow.savefig(save_path_elbow)
        print(f"Saved plot: {save_path_elbow}")
        plt.show()

        optimal_k = k_values[0]
        if len(df_cluster) < optimal_k:
            print(f"WARNING: Number of data points ({len(df_cluster)}) is less than optimal_k ({optimal_k}). Adjusting k.")
            optimal_k = len(df_cluster)

        print(f"Running K-Means with k={optimal_k}...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(df_scaled)

        df_cluster["Cluster"] = cluster_labels
        data = data.merge(
            df_cluster[["Cluster"]], left_index=True, right_index=True, how="left"
        )

        print(f"Generating Plot 10: Customer Segments (k={optimal_k})")
        fig_kmeans = plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=df_cluster[features[0]],
            y=df_cluster[features[1]],
            hue=df_cluster["Cluster"],
            palette="viridis",
            s=100,
        )
        plt.title(f"Customer Segments (k={optimal_k})")
        plt.xlabel(f"{features[0]} (Scaled)")
        plt.ylabel(f"{features[1]} (Scaled)")
        plt.grid(True)
        save_path_kmeans = os.path.join(plots_dir, f"figure_10_customer_segments_k{optimal_k}.png")
        fig_kmeans.savefig(save_path_kmeans)
        print(f"Saved plot: {save_path_kmeans}")
        plt.show()

        print(
            f"\nCluster Statistics for k={optimal_k}:\n",
            data.loc[df_cluster.index].groupby("Cluster")[features].agg(["mean", "median", "count"])
        )

        if optimal_k > 1:
            silhouette_avg = silhouette_score(df_scaled, cluster_labels)
            print(f"Silhouette Score for k={optimal_k}: {silhouette_avg:.3f}")
        else:
            print("Silhouette Score not applicable for k=1.")

    except Exception as e:
        print(f"ERROR during K-Means Clustering: {e}")
        # Removed traceback

    return data


# Run K-means Clustering and update df
df = perform_kmeans_clustering(df, ["Amount", "QtyRequired"])

# === BUSINESS INTERPRETATION ===
print("\n--- Business Interpretation ---")
print("\n" + "=" * 50)
print("BUSINESS INTERPRETATION SUGGESTIONS")
print("=" * 50)
print(
    """
Based on Amount & QtyRequired clustering (k=3):
Cluster 0: Majority, low/medium value & quantity. (Typical Customers)
Cluster 1: Small group, very high value, medium quantity. (VIPs / High-End Buyers)
Cluster 2: Small group, medium value, very high quantity. (Bulk Buyers)

Consider refining segments with RFM metrics for more actionable insights.
"""
)

# Assign meaningful cluster names
cluster_names = {
    0: "Regular Customers",
    1: "High-Value Buyers",
    2: "Bulk Purchasers",
}

# Define functions first
def assign_cluster_names(df, cluster_column, cluster_names):
    """Assign meaningful names to clusters."""
    print("\nAssigning cluster names...")
    if cluster_column in df.columns:
        df[cluster_column] = df[cluster_column].fillna(-1)
        cluster_names_extended = cluster_names.copy()
        cluster_names_extended[-1] = "Unsegmented (Missing Data)"
        df["Segment"] = df[cluster_column].map(cluster_names_extended)
        print(f"Segment value counts:\n{df['Segment'].value_counts()}")
    else:
        print(f"WARNING: '{cluster_column}' column missing. Cannot assign segment names.")
    return df


def plot_cluster_boxplot(df, cluster_column, value_column, title):
    """Plot a boxplot for a given cluster column and value column. Also saves the plot."""
    print(f"\nGenerating Plot 11: {title}")
    if cluster_column in df.columns and value_column in df.columns:
        plot_data = df[df[cluster_column] != -1]
        if not plot_data.empty:
            try:
                fig_box_final = plt.figure(figsize=(10, 6))
                sns.boxplot(x=cluster_column, y=value_column, data=plot_data)
                plt.title(title)
                filename = f"figure_11_{value_column.lower()}_by_{cluster_column.lower()}.png"
                save_path_box_final = os.path.join(plots_dir, filename)
                fig_box_final.savefig(save_path_box_final)
                print(f"Saved plot: {save_path_box_final}")
                plt.show()
            except Exception as e:
                 print(f"ERROR generating/saving Plot 11 ({title}): {e}")
                 # Removed traceback
        else:
            print(f"WARNING: No segmented data found to generate boxplot '{title}'.")

    else:
        print(f"WARNING: Cannot plot boxplot '{title}'. Missing column: '{cluster_column if cluster_column not in df.columns else value_column}'.")


# Use the functions
try:
    df = assign_cluster_names(df, "Cluster", cluster_names)
    plot_cluster_boxplot(
        df,
        cluster_column="Cluster",
        value_column="Amount",
        title="Order Amount Distribution by Cluster",
    )
except Exception as e:
    print(f"ERROR during final business interpretation step: {e}")
    # Removed traceback

print("\n--- Script Finished ---")