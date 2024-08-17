# # Air Traffic Clustering
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/Air_Traffic_Passenger_Statistics.csv')
print(df.head().to_string())
print()

# Check data description
print(df.describe(include='all').to_string())
print()

# # Data Preprocessing
# # Drop unnecessary columns
df.drop(['Activity Period'], axis=1, inplace=True)

# Check missing values
print(df.isnull().sum())
print()

# Drop missing values
df.dropna(inplace=True)

# # Handling Categorical Data using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
mapping = {}
for i in df.columns:
    if df[i].dtypes == 'object':
        df[i] = label_encoder.fit_transform(df[i])
        # add to mapping
        mapping[i] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        # print(mapping[i])

# print()

# Standardizing the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# K-means clustering
# Create a KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df_scaled)

# assign the cluster labels to the original dataframe
df_kmeans = df.copy()
df_kmeans['cluster'] = kmeans.labels_

# Evaluate the model using Silhouette Score
silhouette = silhouette_score(df_scaled, kmeans.labels_)
print('K-means Clustering: Silhouette Score:', silhouette)

# Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering

# Create an AgglomerativeClustering model with 3 clusters
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(df_scaled)

# assign the cluster labels to the original dataframe
df_hiera = df.copy()
df_hiera['cluster'] = agg.labels_

# Evaluate the model using Silhouette Score
silhouette = silhouette_score(df_scaled, agg.labels_)
print('Hierarchical Clustering: Silhouette Score:', silhouette)

# DBSCAN clustering
from sklearn.cluster import DBSCAN

# Create a DBSCAN model
dbscan = DBSCAN(eps=1, min_samples=5)
dbscan.fit(df_scaled)

# assign the cluster labels to the original dataframe
df_dbscan = df.copy()
df_dbscan['cluster'] = dbscan.labels_

# Evaluate the model using Silhouette Score
silhouette = silhouette_score(df_scaled, dbscan.labels_) # DBSCAN provides just 1 cluster, raise error
print('DBSCAN Clustering: Silhouette Score:', silhouette)


# # # Visualize the clustering result # # #
# PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Visualize the data using scatter plot
plt.scatter(df_pca[:, 0], df_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA')
plt.show()

# Visualize the clustering result
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_kmeans['cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering with 3 clusters')
plt.show()

# Visualize the Hierarchical clustering result
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_hiera['cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Hierarchical Clustering with 3 clusters')
plt.show()

# Visualize the clustering result
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_dbscan['cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('DBSCAN Clustering')
plt.show()