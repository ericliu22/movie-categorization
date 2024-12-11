import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Load data from processed.json
with open('../../data/processed.json', 'r') as file:
    data = json.load(file)["data"]

descriptions = [" ".join(entry["description"]) for entry in data]
genres = [entry["genre"] for entry in data]

# Use sklearn's TfidfVectorizer to calculate TF-IDF features
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(descriptions)

# Reduce dimensionality with PCA (TruncatedSVD for sparse matrices)
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X)

# Binarize the genres for multi-label classification
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(genres)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.1, random_state=42)

# Train and evaluate Logistic Regression model with OneVsRestClassifier
lr_classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
lr_classifier.fit(X_train, y_train)

y_pred_lr = lr_classifier.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr, average='micro')
lr_recall = recall_score(y_test, y_pred_lr, average='micro')

print("Logistic Regression Results:")
print(f"Accuracy: {lr_accuracy:.2f}")
print(f"Precision: {lr_precision:.2f}")
print(f"Recall: {lr_recall:.2f}")

# Plot model performance
models = ['Logistic Regression']
accuracies = [lr_accuracy]

plt.bar(models, accuracies, color=['blue'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()