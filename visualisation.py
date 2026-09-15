import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import umap
# Load dataset
digits = load_digits()
data = digits.data
# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_result = tsne.fit_transform(data)
# Apply UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
umap_result = umap_model.fit_transform(data)
# Plot t-SNE result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=digits.target, cmap='Spectral')
plt.title('t-SNE')
# Plot UMAP result
plt.subplot(1, 2, 2)
plt.scatter(umap_result[:, 0], umap_result[:, 1], c=digits.target, cmap='Spectral')
plt.title('UMAP')
plt.show()