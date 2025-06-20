import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading 20 Newsgroups dataset...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
texts = data.data
labels = data.target

# Helper: Load GloVe embeddings
def load_glove_embeddings(glove_path, dim):
    print(f"\nLoading GloVe {dim}d vectors...")
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in tqdm(f, desc=f"Reading {dim}d GloVe"):
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

# Helper: Convert text to averaged word vectors
def text_to_avg_vector(text, embeddings_index, dim):
    words = text.lower().split()
    word_vectors = [embeddings_index[word] for word in words if word in embeddings_index]
    if not word_vectors:
        return np.zeros(dim)
    return np.mean(word_vectors, axis=0)

# Define embedding settings
glove_files = {
    50: 'glove.6B.50d.txt',
    100: 'glove.6B.100d.txt',
    200: 'glove.6B.200d.txt',
    300: 'glove.6B.300d.txt',
}
results = {}

# Main loop: Evaluate each embedding dimension
for dim, path in glove_files.items():
    if not os.path.exists(path):
        print(f"❌ File {path} not found! Please download it from GloVe website.")
        continue

    # Load embeddings
    glove = load_glove_embeddings(path, dim)

    # Vectorize all texts
    print(f"Vectorizing texts with {dim}d embeddings...")
    X_vectors = np.array([text_to_avg_vector(text, glove, dim) for text in tqdm(texts, desc=f"Vectorizing {dim}d")])
    y = np.array(labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    results[dim] = acc
    print(f"✅ Accuracy with {dim}d embeddings: {acc:.4f}")

# Plot results
dims = list(results.keys())
accs = list(results.values())

plt.figure(figsize=(8, 5))
plt.plot(dims, accs, marker='o')
plt.title("Effect of GloVe Embedding Dimension on Accuracy")
plt.xlabel("Embedding Dimension")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(dims)
plt.show()
