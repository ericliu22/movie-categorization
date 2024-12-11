import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import process_data

# Load data from processed.json
with open('../../data/processed.json', 'r') as file:
    data = json.load(file)["data"]

descriptions = [entry["description"] for entry in data]
genres = [entry["genre"] for entry in data]

# Tokenize descriptions using process_data functions
tokenized_descriptions = [process_data.tokenize(" ".join(desc)) for desc in descriptions]

# Calculate IDF scores
idf_scores = process_data.get_idf(tokenized_descriptions)

# Calculate TF-IDF vectors
tfidf_vectors = [
    process_data.calculate_tfidf(desc, idf_scores) for desc in tokenized_descriptions
]

# Convert TF-IDF vectors to a uniform feature matrix
all_terms = list(idf_scores.keys())
term_index = {term: idx for idx, term in enumerate(all_terms)}

X = np.zeros((len(tfidf_vectors), len(all_terms)))
for i, tfidf in enumerate(tfidf_vectors):
    for term, score in tfidf.items():
        if term in term_index:
            X[i, term_index[term]] = score

# Binarize the genres for multi-label classification
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(genres)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train and evaluate KNN classifier
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()