import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk

# Read and preprocess the training data
def read_data(filename):
    sentences = []
    sentence = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '':
                if sentence:
                    sentences.append(sentence)
                sentence = []
            else:
                token, pos_tag, _ = line.split()
                sentence.append((token, pos_tag))
    return sentences


# Extract features from unlabeled data
def extract_unlabeled_features(sentence, index):
    return extract_features([(token, "") for token in sentence], index)

# Feature extraction
# Sentece will be a list of (token, pos_tag) tuples and index will be the position of the token in the sentence
def extract_features(sentence, index):
    tokenDictionary = {} # This is empty dictionary, we will add features to it
    token, _ = sentence[index]

    # Basic features
    tokenDictionary['token'] = token.lower() # Store the lowercase version of the token
    
    # Check the token is given index is the first or last token in the sentence
    # If yes, then store the String <START> to prev_token
    tokenDictionary['prev_token'] = sentence[index - 1][0].lower() if index > 0 else '<START>'
    
    # Check the token is not in the last position of the sentence it is last token then store the String <END> to next_token
    tokenDictionary['next_token'] = sentence[index + 1][0].lower() if index < len(sentence) - 1 else '<END>'
    
    # Enhanced features
    
    # Check the token is all uppercase save the boolean value to is_upper
    tokenDictionary['upper'] = token.isupper()
    
    # Check the token is title case save the boolean value to is_title
    tokenDictionary['title'] = token.istitle()
    
    # Check the token is a digit save the boolean value to is_digit
    tokenDictionary['digit'] = token.isdigit()
    
    # Check the token is a punctuation save suffix
    tokenDictionary['suffix'] = token[-3:]
    
    # Check the token is a punctuation save prefix
    tokenDictionary['prefix'] = token[:3]
    
    # Check the token's length
    tokenDictionary['length'] = len(token)

    return tokenDictionary

# Prepare training data and labels
def ready_for_train(sentences):
    # X stores feature dictionaries and y stores labels
    X = []
    
    # y stores the corresponding part of speech tags
    y = []
    
    # Iterate over sentences and tokens and extract features
    for sentence in sentences:
        
        # Iterate over tokens in the sentence
        for i in range(len(sentence)):
            X.append(extract_features(sentence, i))
            y.append(sentence[i][1])
    return X, y

if __name__ == "__main__":
    # Read the training data
    train_data = read_data('train.txt')

    # Split data into training and development sets
    # train_test_split will split the data into 80% training and 20% development data. Utility function for sklearn
    # 0.2 data will be used for development and 0.8 data will be used for training
    # random_state is used to ensure that the same split is generated every time the code is run
    train_sentences, developing_sentences = train_test_split(train_data, test_size=0.2, random_state=42)

    # Prepare training and development data
    X_train, y_train = ready_for_train(train_sentences)
    X_dev, y_dev = ready_for_train(developing_sentences)

    # Vectorize features
    
    # Use DictVectorizer to convert feature dictionaries into feature vectors
    # Each token's feature were dictionaries it need to convert into feature vectors
    vectorizer = DictVectorizer()
    
    # This learns feature names from training data and assigns an index to each feature in X_train
    X_train = vectorizer.fit_transform(X_train)
    
    # Convert the feature dictionaries in the feature dictionary becomes a column in the matrix
    # If key exist the value will placed in the column otherwise 0 will be placed in the column
    # It will result in a sparse matrix where each row corresponds to a token and each column corresponds to a feature
    X_dev = vectorizer.transform(X_dev)

    # Hyperparameter tuning for Bayesian Classifier
    
    # it contains that specifies potential values for the hyperparameters to use in Multinomial Naive Bayes
    # Navie hyperparameter 'alpha' is used to control the smoothing of the model. 
    # We have to use smoothing because some of the features may not be present in the training data
    # Laplace smoothing adds a small value 'alpha' to the count of each feature
    # When alpha = 1 it is called Laplace smoothing increase alpha value will increase the smoothing
    # When alpha = 0 it is no smoothing is applied to the model increased by a fraction of the total number of features
    # When alpha < 1 it is called Lidstone smoothing
    # Using multiple value of alpha and find the best value of alpha making model more accurate
    # Smaller alpha value will result in more smoothing
    # Larger alpha value will result in less smoothing
    alpha_paramters = {'alpha': [0.001, 0.2, 0.25, 1, 9, 20]}
    
    # Using GridSearchCV to find the best value of alpha this model performance will find the highest 
    # MultiNomialNB is the classifier we are using
    # cv is the number of folds to use in cross-validation repeat the process 5 times
    bayesian_classifier = GridSearchCV(MultinomialNB(), alpha_paramters, cv=5, scoring='f1_weighted')
    
    # For each combination of hyperparameters the model is trained and using cross-validation
    bayesian_classifier.fit(X_train, y_train)

    # Evaluate the best Bayesian Classifier
    # After the GridSearchCV has been fitted with data, it provides several attributes that can be used to evaluate the model
    # best_estimator_ is the best model found by GridSearchCV
    # best_model will return the best model from the grid search
    best_model = bayesian_classifier.best_estimator_
    
    #Find the best value of alpha for the modelmake predictions on the development data
    y_pred = best_model.predict(X_dev)
    
    # This is function from sklearn.metrics to calculate the accuracy of the model the formula accuracy_score(y_true, y_pred) y is the true label and y_pred is the predicted label
    accuracy = accuracy_score(y_dev, y_pred)
    print(f'Bayesian Classifier Accuracy: {accuracy:.4f}')
    
    
    # Read the unlabeled test data
    with open('unlabeled_test_test.txt', 'r') as file:
        unlabeled_data = file.read().split()

    # Extract features from the unlabeled data
    X_unlabeled = [extract_unlabeled_features(unlabeled_data, i) for i in range(len(unlabeled_data))]

    # Convert the feature dictionaries to feature vectors
    X_unlabeled = vectorizer.transform(X_unlabeled)

    # Predict using the trained model
    y_unlabeled_pred = best_model.predict(X_unlabeled)

    # Write the predictions to a new file
    with open('Megabyte_KN.txt', 'w') as file:
        for token, tag in zip(unlabeled_data, y_unlabeled_pred):
            file.write(f"{token} {tag}\n")

    print("Predictions written to tagged_test.txt")