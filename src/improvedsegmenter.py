import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === Setup and paths ===
DATA_DIR = 'data'
FILE_NAME = 'Online Retail.xlsx'
file_path = os.path.join(DATA_DIR, FILE_NAME)

# Make sure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Debug: Print current working directory
print("Current working directory:", os.getcwd())

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

# === Load and clean data ===
data = pd.read_excel(file_path)
data.drop_duplicates(inplace=True)
data.dropna(subset=['CustomerID'], inplace=True)

# Calculate total spent per transaction
data['total_spent'] = data['UnitPrice'] * data['Quantity']

# Remove negative or zero total_spent (e.g., refunds)
data = data[data['total_spent'] > 0]

# Convert date and calculate Recency
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
reference_date = data['InvoiceDate'].max()

# === Build RFM Features ===
rfm = data.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('total_spent', 'sum')
).reset_index()

# === Standardize Data ===
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# === Determine Optimal Clusters using Elbow & Silhouette ===
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(rfm_scaled)
    inertia.append(model.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, labels))

# Plot Elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.tight_layout()
elbow_path = os.path.join(DATA_DIR, 'elbow_plot.png')
plt.savefig(elbow_path)
print(f"Elbow plot saved to {elbow_path}")
plt.close()

# Plot Silhouette scores
plt.figure(figsize=(8, 4))
plt.plot(K_range, silhouette_scores, marker='o', color='green')
plt.title('Silhouette Scores For Different k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.tight_layout()
silhouette_path = os.path.join(DATA_DIR, 'silhouette_plot.png')
plt.savefig(silhouette_path)
print(f"Silhouette plot saved to {silhouette_path}")
plt.close()

# === Apply Final KMeans ===
optimal_k = 4  # Chosen based on elbow and silhouette
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

# Save centroids in original scale
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)

# === Segment Summary ===
summary = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(2)

print("Segment Summary:\n", summary)

# === Visualize Customer Segments ===
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=rfm,
    x='Recency',
    y='Monetary',
    hue='Segment',
    palette='viridis',
    s=100,
    edgecolor='k'
)

# Plot centroids
plt.scatter(
    centroids[:, 0],  # Recency
    centroids[:, 2],  # Monetary
    c='red',
    s=300,
    marker='X',
    label='Centroids'
)

plt.title('Customer Segments Based on RFM')
plt.xlabel('Recency (Days Since Last Purchase)')
plt.ylabel('Monetary (Total Spend)')
plt.legend()
plt.tight_layout()

segment_plot_path = os.path.join(DATA_DIR, 'rfm_segmentation.png')
plt.savefig(segment_plot_path)
print(f"Segment plot saved to {segment_plot_path}")
plt.close()

# === Save Output CSV ===
output_path = os.path.join(DATA_DIR, 'customers_segmented.csv')
rfm.to_csv(output_path, index=False)
print(f"Segmentation complete. Results saved to: {output_path}")