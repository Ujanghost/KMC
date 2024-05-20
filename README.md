# K-Means Clustering on WhatsApp Messages

This document explains the Python code used to convert WhatsApp chat data into a structured CSV format, preprocess the text data, apply K-Means clustering, and visualize the clustering results. The clustering is performed with different values of 'k' (from 2 to 5) to analyze the clusters formed.

## Libraries Required

Ensure the following libraries are installed:

```sh
pip install numpy pandas scikit-learn matplotlib
```

## Code Explanation

### 1. Convert WhatsApp Text Data to CSV

```python
import pandas as pd

# Read the WhatsApp text data
with open('whatsapp.txt', 'r') as file:
    lines = file.readlines()

# Parse the lines and create a structured DataFrame
data = []
for line in lines:
    parts = line.split(']', 1)
    if len(parts) == 2:
        date_time = parts[0].strip('[')
        message = parts[1].split(':', 1)
        if len(message) == 2:
            author = message[0].strip()
            text = message[1].strip()
            data.append([date_time, author, text])

# Create a DataFrame
df = pd.DataFrame(data, columns=['DateTime', 'Author', 'Message'])

# Save the DataFrame to a CSV file
df.to_csv('whatsapp.csv', index=False)
```

- **Data Loading**: The WhatsApp text data is read line by line.
- **Data Parsing**: Each line is split to extract the date, time, author, and message.
- **DataFrame Creation**: The parsed data is stored in a pandas DataFrame.
- **CSV Export**: The DataFrame is saved to a CSV file.

### 2. Preprocess Text Data and Apply K-Means Clustering

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the CSV file
df = pd.read_csv('whatsapp.csv')

# Extract the messages
messages = df['Message'].tolist()

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(messages)

# Function to perform K-Means clustering and visualize the results
def kmeans_clustering(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Visualize the clusters using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    plt.figure(figsize=(8, 6))
    for i in range(k):
        plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i+1}')
    plt.title(f'K-Means Clustering with k={k}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

    # Visualize the clusters using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X.toarray())

    plt.figure(figsize=(8, 6))
    for i in range(k):
        plt.scatter(X_tsne[labels == i, 0], X_tsne[labels == i, 1], label=f'Cluster {i+1}')
    plt.title(f'K-Means Clustering with k={k} (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.show()

    # Print the cluster analysis
    df['Cluster'] = labels
    for i in range(k):
        print(f'\nCluster {i+1} Analysis:')
        cluster_messages = df[df['Cluster'] == i]['Message']
        print(cluster_messages.value_counts().head())

# Apply K-Means clustering and visualize results for k=2 to k=5
for k in range(2, 6):
    kmeans_clustering(X, k)
```

#### Detailed Explanation

1. **Load the CSV File**:
    - The WhatsApp CSV file created earlier is loaded into a pandas DataFrame.

2. **Extract and Preprocess Messages**:
    - Extract the 'Message' column from the DataFrame.
    - Convert the text messages into TF-IDF features using `TfidfVectorizer` to transform the text data into numerical format suitable for clustering.

3. **K-Means Clustering Function**:
    - Define a function `kmeans_clustering` to perform K-Means clustering with a specified number of clusters `k`.
    - **Training**: Train the K-Means model on the TF-IDF features.
    - **PCA Visualization**: Use Principal Component Analysis (PCA) to reduce the dimensions of the TF-IDF features to 2D and plot the clusters.
    - **t-SNE Visualization**: Use t-Distributed Stochastic Neighbor Embedding (t-SNE) for another form of 2D visualization of the clusters.
    - **Cluster Analysis**: Assign the cluster labels to the DataFrame and print the top messages in each cluster for analysis.

4. **Apply K-Means Clustering for k=2 to k=5**:
    - Apply the `kmeans_clustering` function for `k` values from 2 to 5.
    - Visualize and analyze the clusters formed for each value of `k`.

## Conclusion

This code provides a comprehensive approach to clustering WhatsApp messages using K-Means. It involves converting the raw text data into a structured format, preprocessing the text data to extract meaningful features, applying K-Means clustering with varying values of `k`, and visualizing the clustering results using PCA and t-SNE. The analysis of cluster content helps in understanding the distribution and common themes within each cluster.

---
