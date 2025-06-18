import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------
# Step 1: Load Dataset
# -------------------------------------
# Set path to dataset (relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "Mall_Customers.csv")

# Read CSV
data = pd.read_csv(data_path)

print("First 5 rows of the dataset:")
print(data.head())

# -------------------------------------
# Step 2: Data Preprocessing
# -------------------------------------

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Select numerical features for clustering
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------
# Step 3: Elbow Method to Find Optimal K
# -------------------------------------

# Ensure 'visuals' folder exists
visuals_dir = os.path.join(script_dir, "..", "visuals")
os.makedirs(visuals_dir, exist_ok=True)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)

elbow_path = os.path.join(visuals_dir, "elbow_plot.png")
plt.savefig(elbow_path)
print(f"\nElbow plot saved to: {elbow_path}")
plt.close()

# -------------------------------------
# Step 4: K-Means Clustering with Optimal K
# -------------------------------------

optimal_k = 5  # Based on Elbow Plot (manual inspection)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to original data
data["Cluster"] = clusters

# -------------------------------------
# Step 5: Visualize the Clusters
# -------------------------------------

plt.figure(figsize=(8, 5))
colors = ['red', 'green', 'blue', 'cyan', 'magenta']

for i in range(optimal_k):
    plt.scatter(
        X_scaled[clusters == i, 0], 
        X_scaled[clusters == i, 1], 
        s=100, 
        c=colors[i], 
        label=f'Cluster {i}'
    )

plt.title('Customer Segments')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()

clusters_path = os.path.join(visuals_dir, "clusters_plot.png")
plt.savefig(clusters_path)
print(f"Cluster plot saved to: {clusters_path}")
plt.close()
