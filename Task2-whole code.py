import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 加载数据
print("Loading dataset...")
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
texts = newsgroups.data
labels = newsgroups.target
target_names = newsgroups.target_names

# 2. 加载 GloVe 词向量
def load_glove_embeddings(glove_path='glove.6B.100d.txt'):
    print("Loading GloVe vectors...")
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in tqdm(f, desc="Reading GloVe file"):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index

glove_path = 'glove.6B.100d.txt'
if not os.path.exists(glove_path):
    raise FileNotFoundError(f"{glove_path} not found. Please download it from https://nlp.stanford.edu/projects/glove/")
embeddings_index = load_glove_embeddings(glove_path)

# 3. 文本转向量
def text_to_avg_vector(text, embeddings_index, embedding_dim=100):
    words = text.lower().split()
    word_vectors = [embeddings_index[word] for word in words if word in embeddings_index]
    if not word_vectors:
        return np.zeros(embedding_dim)
    return np.mean(word_vectors, axis=0)

print("Vectorizing texts with GloVe embeddings...")
X_glove = np.array([text_to_avg_vector(text, embeddings_index) for text in tqdm(texts, desc="Text to vectors")])
y = np.array(labels)

# 4. 划分数据
X_train_glove, X_test_glove, y_train, y_test = train_test_split(X_glove, y, test_size=0.2, random_state=42)

# 5. GloVe 模型训练
print("Training classifier with GloVe embeddings...")
clf_glove = LogisticRegression(max_iter=1000)
clf_glove.fit(X_train_glove, y_train)
y_pred_glove = clf_glove.predict(X_test_glove)
acc_glove = accuracy_score(y_test, y_pred_glove)
print(f"\nAccuracy using GloVe: {acc_glove:.4f}")
print(classification_report(y_test, y_pred_glove, target_names=target_names))

# 6. TF-IDF 特征对比
print("Generating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = vectorizer.fit_transform(texts)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

print("Training classifier with TF-IDF features...")
clf_tfidf = LogisticRegression(max_iter=1000)
clf_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
print(f"\nAccuracy using TF-IDF: {acc_tfidf:.4f}")
print(classification_report(y_test, y_pred_tfidf, target_names=target_names))

# 7. 绘图比较准确率
plt.figure(figsize=(6, 4))
plt.bar(['GloVe', 'TF-IDF'], [acc_glove, acc_tfidf], color=['skyblue', 'salmon'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 8. 混淆矩阵可视化（GloVe）
cm = confusion_matrix(y_test, y_pred_glove)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - GloVe Embeddings')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

