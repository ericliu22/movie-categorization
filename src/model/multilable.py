import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score
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

# Custom accuracy function for multi-label classification
def custom_accuracy(y_true, y_pred):
    correct_predictions = 0
    for true_labels, pred_labels in zip(y_true, y_pred):
        true_labels_set = set(np.where(true_labels == 1)[0])
        pred_labels_set = set(np.where(pred_labels == 1)[0])
        if true_labels_set & pred_labels_set:
            correct_predictions += 1
    return correct_predictions / len(y_true)

# Train and evaluate Multi-label KNN classifier
neighbors = np.arange(1, 9)
traditional_accuracies = []
custom_accuracies = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on test data
    y_test_pred = knn.predict(X_test)

    # Calculate traditional accuracy
    traditional_accuracy = accuracy_score(y_test, y_test_pred)
    traditional_accuracies.append(traditional_accuracy)

    # Calculate custom accuracy
    custom_acc = custom_accuracy(y_test, y_test_pred)
    custom_accuracies.append(custom_acc)

# Generate plot for traditional and custom accuracy
plt.figure(figsize=(10, 6))
plt.plot(neighbors, traditional_accuracies, label='Traditional Accuracy')
plt.plot(neighbors, custom_accuracies, label='Custom Accuracy')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('Multi-label KNN Accuracy')
plt.grid()
plt.show()

# Evaluate best model
best_k = neighbors[np.argmax(custom_accuracies)]
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Calculate final metrics
traditional_accuracy = accuracy_score(y_test, y_pred)
custom_acc = custom_accuracy(y_test, y_pred)

print(f"Best Multi-label KNN Model (k={best_k}):")
print(f"Traditional Accuracy: {traditional_accuracy * 100:.2f}%")
print(f"Custom Accuracy: {custom_acc * 100:.2f}%")

# Print first 10 predictions and true labels
print("First 10 Predictions:")
print(y_pred[:10])
print("First 10 True Labels:")
print(y_test[:10])
