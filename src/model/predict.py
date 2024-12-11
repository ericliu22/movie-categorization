import json
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from process_data import tokenize, get_idf, calculate_tfidf

# Load processed data
with open('../../data/processed.json', 'r') as f:
    data = json.load(f)['data']

# Extract descriptions and genres
descriptions = [item['description'] for item in data]
genres = [item['genre'] for item in data]

# Tokenize descriptions
tokenized_descriptions = [tokenize(' '.join(desc)) for desc in descriptions]

# Calculate IDF scores
idf_scores = get_idf(tokenized_descriptions)

# Calculate TF-IDF for each description
tfidf_descriptions = [calculate_tfidf(tokens, idf_scores) for tokens in tokenized_descriptions]

# Convert TF-IDF to feature vectors
unique_terms = list(idf_scores.keys())
def tfidf_to_vector(tfidf, terms):
    return np.array([tfidf.get(term, 0) for term in terms])

feature_vectors = np.array([tfidf_to_vector(tfidf, unique_terms) for tfidf in tfidf_descriptions])

# Encode genres
unique_genres = list(set(genre for sublist in genres for genre in sublist))
def encode_genres(genre_list, unique_genres):
    return np.array([1 if genre in genre_list else 0 for genre in unique_genres])

genre_vectors = np.array([encode_genres(genre_list, unique_genres) for genre_list in genres])

# Split data
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, genre_vectors, test_size=0.2, random_state=42)

# Custom accuracy function for multi-label classification
def custom_accuracy(y_true, y_pred):
    correct_predictions = 0
    for true_labels, pred_labels in zip(y_true, y_pred):
        true_labels_set = set(np.where(true_labels == 1)[0])
        pred_labels_set = set(np.where(pred_labels == 1)[0])
        if true_labels_set & pred_labels_set:
            correct_predictions += 1
    return correct_predictions / len(y_true)

# Loop over K values to calculate accuracy
neighbors = np.arange(1, 11)
traditional_accuracies = []
custom_accuracies = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Calculate traditional accuracy
    traditional_accuracy = accuracy_score(y_test, y_pred.round())
    traditional_accuracies.append(traditional_accuracy)

    # Calculate custom accuracy
    custom_acc = custom_accuracy(y_test, y_pred.round())
    custom_accuracies.append(custom_acc)

# Plot accuracies
plt.figure(figsize=(10, 6))
plt.plot(neighbors, traditional_accuracies, label='Traditional Accuracy')
plt.plot(neighbors, custom_accuracies, label='Custom Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Neighbors')
plt.legend()
plt.grid()
plt.show()

# Print final results for k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
traditional_accuracy = accuracy_score(y_test, y_pred.round())
custom_acc = custom_accuracy(y_test, y_pred.round())
print(f"Traditional Accuracy (k=5): {traditional_accuracy * 100:.2f}%")
print(f"Custom Accuracy (k=5): {custom_acc * 100:.2f}%")

# Print first 10 predictions and true labels
print("First 10 Predictions:")
print(y_pred[:10])
print("First 10 True Labels:")
print(y_test[:10])
