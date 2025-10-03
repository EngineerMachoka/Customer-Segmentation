import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up path to data (relative to project root)
file_path = os.path.join('data', 'Online Retail.xlsx')

# Load Dataset
data = pd.read_excel(file_path)

# Clean Data: remove duplicates and rows with missing CustomerID
data = data.drop_duplicates()
data = data.dropna(subset=['CustomerID'])

# Calculate Total Spent per transaction
data['total_spent'] = data['UnitPrice'] * data['Quantity']

# Aggregate Total Spent and Quantity using Customer ID
customer_data = data.groupby('CustomerID').agg({
    'total_spent': 'sum',
    'Quantity': 'sum'
}).reset_index()

# Standardize data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[['total_spent', 'Quantity']])

# KMeans Clustering
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
customer_data['Segment'] = kmeans.fit_predict(customer_data_scaled)

# Reverse transform cluster centres to original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Visualization
sns.set()
plt.figure(figsize=(10, 6))

# Plot customer segments
sns.scatterplot(
    data=customer_data,
    x='Quantity',
    y='total_spent',
    hue='Segment',
    palette='viridis',
    s=100,
    edgecolor='k'
)

# Plot cluster centroids
plt.scatter(
    centroids[:, 1],  # Quantity (x-axis)
    centroids[:, 0],  # Total Spent (y-axis)
    s=300,
    c='red',
    marker='X',
    label='Centroids'
)

plt.title('Customer Segments based on Quantity and Total Spent')
plt.xlabel('Quantity')
plt.ylabel('Total Spent')
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join('data', 'segmentation_plot.png'))

# Optionally show the plot
plt.show()

# Save the customer segmentation results
customer_data.to_csv(os.path.join('data', 'customers_segmented.csv'), index=False)

print("✅ Segmentation complete. Results and plot saved in /data folder.")
