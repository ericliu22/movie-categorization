import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

# Load data from processed.json
with open('../../data/processed.json', 'r') as file:
    data = json.load(file)["data"]

descriptions = [" ".join(entry["description"]) for entry in data]
genres = [entry["genre"] for entry in data]

# Use sklearn's TfidfVectorizer to calculate TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descriptions)

# Normalize features using StandardScaler
scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrices
X_scaled = scaler.fit_transform(X)

# Binarize the genres for multi-label classification
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(genres)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Train and evaluate Multi-label KNN classifier
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on training and test data
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    # Compute accuracy for training and test sets
    train_accuracy[i] = accuracy_score(y_train, y_train_pred)
    test_accuracy[i] = accuracy_score(y_test, y_test_pred)

# Generate plot
plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('Multi-label KNN Accuracy')
plt.show()

# Evaluate best model
best_k = neighbors[np.argmax(test_accuracy)]
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Calculate overall metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')

print(f"Best Multi-label KNN Model (k={best_k}):")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

