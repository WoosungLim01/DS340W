import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC  # Import SVM classifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk

# Constants
# train_test_split will split the data into 80% training and 20% development data. Utility function for sklearn
# 0.2 data will be used for development and 0.8 data will be used for training
test_size = 0.4
random_state = 42
file = 'train.txt'

# Read and preprocess the training data
# List to store sentences,  list to store words within each sentence
# Remove leading/trailing whitespace
# Split the line into token, POS tag, and discard the third part
# Append (token, POS tag) tuple to the word list
def read_file(filename):
    sent = []  
    word = [] 
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip() 
            if line == '':
                if word:
                    sent.append(word)  
                word = []  
            else:
                
                token, pos_tag, _ = line.split()
                word.append((token, pos_tag))  
    return sent


def extract_unlabeled_features(sentence, index):
    return feature([(token, "") for token in sentence], index)

# Feature extraction
# Get the token and POS tag from the sentence
def feature(sent, num):
    features = {}
    token, _ = sent[num]

    # Basic features
    features['token'] = token.lower()  # Convert token to lowercase
    features['token_prev'] = sent[num - 1][0].lower() if num > 0 else '<START>'  # Token from the previous position
    features['token_next'] = sent[num + 1][0].lower() if num < len(sent) - 1 else '<END>'  # Token from the next position

    # Enhanced features
    features['upper'] = token.isupper()  # Token is in uppercase
    features['title'] = token.istitle()  # Token is in title case
    features['digit'] = token.isdigit()  # Token consists of digits
    features['suffix'] = token[-3:]  # Last 3 characters of the token
    features['prefix'] = token[:3]  # First 3 characters of the token
    features['length'] = len(token)  # Length of the token
    features['2gram'] = token[-2:]  # Last 2 characters of the token
    features['3gram'] = token[-3:]  # Last 3 characters of the token
    features['hyphen'] = '-' in token  # Token contains a hyphen
    features['alphanumeric'] = token.isalnum()  # Token is alphanumeric

    return features

# Prepare training data and labels
def data_train(sentences):
    X = []  # List to store feature dictionaries
    y = []  # List to store POS tags
    for sentence in sentences:
        for i in range(len(sentence)):
            X.append(feature(sentence, i))  # Extract features for each word in the sentence
            y.append(sentence[i][1])  # Append the POS tag for each word
    return X, y

if __name__ == "__main__":

    # Read the training data
    train_data = read_file(file)

    # Split data into training and development sets
    train_sentences, dev_sentences = train_test_split(train_data, test_size, random_state)

    # Prepare training and development data
    X_trained_data, y_trained_data = data_train(train_sentences)
    X_developed, y_developed = data_train(dev_sentences)

    # Vectorize features using DictVectorizer
    vectorizer = DictVectorizer()
    X_trained_data = vectorizer.fit_transform(X_trained_data)
    X_developed = vectorizer.transform(X_developed)

    # Hyperparameter tuning for SVM
    param = {'C': [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    svm = GridSearchCV(SVC(), param, cv=5, scoring='f1_weighted')  # Grid search with cross-validation
    svm.fit(X_trained_data, y_trained_data)  # Fit the SVM model

    # Evaluate the best SVM model on the development set
    model = svm.best_estimator_
    y_pred = model.predict(X_developed)
    acc = accuracy_score(y_developed, y_pred)  # Calculate accuracy
    print(f'SVM Accuracy: {acc:.4f}')
    
    # Read the unlabeled test data
    with open('unlabeled_test_test.txt', 'r') as file:
        unlabeled_data = file.read().split()

    # Extract features from the unlabeled data
    X_unlabeled = [extract_unlabeled_features(unlabeled_data, i) for i in range(len(unlabeled_data))]

    # Convert the feature dictionaries to feature vectors
    X_unlabeled = vectorizer.transform(X_unlabeled)

    # Predict using the trained model
    y_unlabeled_pred = model.predict(X_unlabeled)

    # Write the predictions to a new file
    with open('Megabyte_KN.txt', 'w') as file:
        for token, tag in zip(unlabeled_data, y_unlabeled_pred):
            file.write(f"{token} {tag}\n")

    print("Predictions written to tagged_test.txt")
